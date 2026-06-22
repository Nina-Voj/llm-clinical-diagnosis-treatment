# ============================================================
# BSP2 / Project Omega -- Step 1: Main Pipeline
# 
# RUN ORDER:
# 1. patients_pipeline.py    <-- THIS FILE (Generates LLM responses & evaluator scores)
# 2. bleu_rouge_metrics.py      (Calculates text similarity metrics)
# 3. temperature_test.py             (Runs the hyperparameter experiment)
# 4. graph_maker.py                  (Builds the main visualization dashboard)
# 5. lime_analyzer.py           (Runs LIME explainability analysis)
# ============================================================
import os
import duckdb
import re
import json
import time
import difflib
import pandas as pd
# openai imported later in the API section after key validation
from colorama import init, Fore, Style
init(autoreset=True)

# =====================================================================
# 1. CONFIGURATION
# Here we define the main settings for our pipeline: which LLM models to test,
# how many patients to process, and API rate limit safeguards.
# =====================================================================
MIMIC_DB          = r"C:\Your mimic.db adress" # Local indexed database(EDIT THIS)
MAX_PATIENTS      = 4000  # Set to a number to limit, or None to process all patients
RANDOM_SEED       = 42    # Fixed seed for reproducible random patient sampling (set a seed)
BATCH_SIZE        = 50    # Number of patients to load into memory at once (keeps RAM usage low)
SLEEP_BETWEEN     = 1     # seconds between model calls (API rate limit buffer)
SLEEP_BETWEEN_PAT = 1     # seconds between patients (API rate limit buffer)
RUN_EVALUATOR     = True
EVAL_ONLY_FIRST_N = None  # None = evaluate all 4000 patients (no API cost on local HPC)
OUTPUT_CSV        = "mimic_batch_results.csv"
EVAL_CSV          = "mimic_batch_evaluations.csv"



MODEL1_NAME    = "llama-3.3-70b-versatile" # Groq -> OpenRouter fallback
MODEL2_NAME    = "gemma-4-31b-it" # Google Gemini API
EVALUATOR_NAME = "openai/gpt-oss-120b" # Groq -> OpenRouter fallback
FUZZY_THRESHOLD = 0.75 # Similarity threshold (75%) to count a recommended drug as a match
PRECISION_K     = 5    # Max number of recommended drugs evaluated for Precision@K
TEMPERATURE     = 0.3  # moderate temperature for more creative outputs
MAX_TOKENS      = 2500 # limits response length

# =====================================================================
# 2. DATA LOADING (DuckDB)
# For large cohorts (tens of thousands of patients) we use a two-phase
# loading strategy:
#   Phase 1 – get_patient_pool(): lightweight query returning only patient
#              IDs, demographics, and diagnoses. Runs once at startup.
#   Phase 2 – get_batch_data(): heavy query loading labs, prescriptions,
#              vitals, etc. for a small BATCH of patients at a time.
#              Memory is released between batches.
# =====================================================================


COHORT_CSV        = "cohort_ids.csv"  # persisted patient list — same across all runs

def get_patient_pool(db_path, max_patients=MAX_PATIENTS, seed=RANDOM_SEED):
    """
    Phase 1: Lightweight query — returns only patient IDs, demographics,
    and diagnoses. Safe to load for any cohort size (rows are small).

    Patient selection strategy (reproducible random sampling):
      - If cohort_ids.csv exists, load it directly — guarantees the exact same
        patients every run, even after a crash or on a different machine.
      - If it does not exist, draw all eligible hadm_ids, shuffle with Python's
        seeded random, take the first max_patients, and persist to CSV.

    This means the resume system (done_ids) correctly skips already-processed
    patients regardless of how many times the script is restarted.
    """
    print(Fore.CYAN + "[load] connecting to DB and fetching patient pool...")
    con = duckdb.connect(db_path, read_only=True)

    if os.path.exists(COHORT_CSV):
        saved = pd.read_csv(COHORT_CSV)
        all_ids = saved["hadm_id"].tolist()
        print(Fore.YELLOW + f"[load] cohort loaded from {COHORT_CSV} ({len(all_ids)} patients available).")
    else:
        # Fetch ALL eligible hadm_ids (no LIMIT)
        all_ids_df = con.query("""
            SELECT DISTINCT a.hadm_id
            FROM admissions a
            JOIN patients p ON a.subject_id = p.subject_id
            WHERE a.hadm_id IN (SELECT DISTINCT hadm_id FROM diagnoses_icd)
              AND a.hadm_id IN (SELECT DISTINCT hadm_id FROM labevents)
              AND a.hadm_id IN (SELECT DISTINCT hadm_id FROM prescriptions)
        """).df()
        all_ids = all_ids_df["hadm_id"].tolist()

        # Reproducible shuffle using Python's seeded random
        import random as _rnd
        rng = _rnd.Random(seed)
        rng.shuffle(all_ids)

        pd.DataFrame({"hadm_id": all_ids}).to_csv(COHORT_CSV, index=False)
        print(Fore.GREEN + f"[load] entire shuffled cohort saved -> {COHORT_CSV} (seed={seed})")

    # Slice the required number of patients for this run
    hadm_ids_ordered = all_ids[:max_patients] if max_patients else all_ids
    print(Fore.CYAN + f"[load] Selecting top {len(hadm_ids_ordered)} patients for this run.")

    # Fetch demographics + diagnoses for the selected cohort only
    ids_sql = str(tuple(hadm_ids_ordered)) if len(hadm_ids_ordered) > 1 else f"({hadm_ids_ordered[0]})"
    query_pool = f"""
    WITH adm_pool AS (
        SELECT a.subject_id, a.hadm_id, p.gender, p.anchor_age
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.hadm_id IN {ids_sql}
    ),
    dx_all AS (
        SELECT dx.hadm_id, dx.seq_num, dx.icd_code, dicd.long_title
        FROM diagnoses_icd dx
        LEFT JOIN d_icd_diagnoses dicd
               ON dx.icd_code = dicd.icd_code AND dx.icd_version = dicd.icd_version
    )
    SELECT p.*, d.icd_code, d.long_title, d.seq_num
    FROM adm_pool p
    LEFT JOIN dx_all d ON p.hadm_id = d.hadm_id
    ORDER BY p.hadm_id, d.seq_num
    """
    pool_df = con.query(query_pool).df()
    con.close()
    print(Fore.GREEN + f"[load] pool ready: {pool_df['hadm_id'].nunique()} unique admissions.")
    return pool_df


def get_batch_data(db_path, batch_pool_df):
    """
    Phase 2: Heavy query — loads labs, prescriptions, vitals, OMR,
    procedures, and microbiology for a SMALL batch of patients only.
    Call this inside the main loop and discard the results after each batch
    to keep RAM usage flat regardless of total cohort size.
    """
    con = duckdb.connect(db_path, read_only=True)

    hadm_ids   = batch_pool_df["hadm_id"].unique().tolist()
    subj_ids   = batch_pool_df["subject_id"].unique().tolist()

    # DuckDB supports passing Python lists directly as parameters
    def ids_sql(lst):
        return f"({lst[0]})" if len(lst) == 1 else str(tuple(lst))

    h_sql = ids_sql(hadm_ids)
    s_sql = ids_sql(subj_ids)

    labs_df = con.query(f"""
        SELECT l.hadm_id, l.charttime, li.label, l.valuenum, l.valueuom, l.flag
        FROM labevents l
        JOIN d_labitems li ON l.itemid = li.itemid
        WHERE l.hadm_id IN {h_sql} AND l.valuenum IS NOT NULL
        ORDER BY l.hadm_id, l.charttime
    """).df()

    rx_df = con.query(
        f"SELECT * FROM prescriptions WHERE hadm_id IN {h_sql}"
    ).df()

    icu_df = con.query(
        f"SELECT * FROM icustays WHERE hadm_id IN {h_sql}"
    ).df()

    vitals_df = con.query(f"""
        SELECT c.hadm_id, c.charttime, di.label, c.valuenum, c.valueuom
        FROM chartevents c
        JOIN d_items di ON c.itemid = di.itemid
        WHERE c.hadm_id IN {h_sql} AND c.valuenum IS NOT NULL
        ORDER BY c.hadm_id, c.charttime
    """).df()

    omr_df = con.query(f"""
        SELECT subject_id, chartdate, result_name, result_value
        FROM omr
        WHERE subject_id IN {s_sql}
        ORDER BY subject_id, chartdate
    """).df()

    proc_df = con.query(f"""
        SELECT pr.hadm_id, pr.seq_num, dp.long_title
        FROM procedures_icd pr
        LEFT JOIN d_icd_procedures dp
               ON pr.icd_code = dp.icd_code AND pr.icd_version = dp.icd_version
        WHERE pr.hadm_id IN {h_sql}
        ORDER BY pr.hadm_id, pr.seq_num
    """).df()

    micro_df = con.query(f"""
        SELECT hadm_id, chartdate, spec_type_desc, org_name, ab_name, interpretation
        FROM microbiologyevents
        WHERE hadm_id IN {h_sql}
        ORDER BY hadm_id, chartdate
    """).df()

    con.close()
    return labs_df, rx_df, icu_df, vitals_df, omr_df, proc_df, micro_df


# ============== COHORT DEMOGRAPHIC SUMMARY =====================
ICD10_CHAPTERS = [
    ("A", "Infectious & parasitic"),
    ("B", "Infectious & parasitic"),
    ("C", "Neoplasms"),
    ("D", "Blood / neoplasms"),
    ("E", "Endocrine / metabolic"),
    ("F", "Mental / behavioural"),
    ("G", "Nervous system"),
    ("H", "Eye / Ear"),
    ("I", "Circulatory"),
    ("J", "Respiratory"),
    ("K", "Digestive"),
    ("L", "Skin"),
    ("M", "Musculoskeletal"),
    ("N", "Genitourinary"),
    ("O", "Pregnancy"),
    ("P", "Perinatal"),
    ("Q", "Congenital"),
    ("R", "Symptoms / signs"),
    ("S", "Injury / poisoning"),
    ("T", "Injury / poisoning"),
    ("Z", "Factors influencing health"),
]
_CHAPTER_MAP = dict(ICD10_CHAPTERS)


def icd_chapter(code):
    if not isinstance(code, str) or len(code) == 0:
        return "unknown"
    first = code[0].upper()
    return _CHAPTER_MAP.get(first, "ICD-9 / other")


def print_cohort_summary(pool):
    # Deduplicate by hadm_id so stats are per-patient, not per-diagnosis
    p = pool.drop_duplicates(subset="hadm_id")
    print(Fore.MAGENTA + Style.BRIGHT + "\n=== COHORT SUMMARY ===")
    ages = p["anchor_age"]
    print(f"  N admissions sampled : {len(p)}")
    print(f"  Age  mean / std      : {ages.mean():.1f} / {ages.std():.1f}")
    print(f"  Age  min / max       : {int(ages.min())} / {int(ages.max())}")
    print(f"  Gender balance       : {p['gender'].value_counts().to_dict()}")
    chapters = p["icd_code"].apply(icd_chapter).value_counts().head(5)
    print("  Top-5 ICD chapters   :")
    for ch, n in chapters.items():
        print(f"     - {ch:<30s} {n}")
    print()


# =====================================================================
# 3. PROMPT CONSTRUCTION
# This function takes the raw clinical data (labs, drugs, etc.) for a patient
# and formats it into a structured prompt for the LLMs to read.
# =====================================================================
def build_prompt(row, pool_df, labs_df, rx_df, icu_df, vitals_df, omr_df, proc_df, micro_df):
    """Construct a richly-structured clinical prompt for one admission."""
    hadm = row["hadm_id"]
    demo = f"Age: {row['anchor_age']} | Gender: {'male' if row['gender'] == 'M' else 'female'}"

    # --- All diagnoses for this admission ---
    dx_rows = pool_df[pool_df["hadm_id"] == hadm][["seq_num", "long_title"]].dropna(subset=["long_title"])
    if not dx_rows.empty:
        dx_lines = []
        for _, r in dx_rows.iterrows():
            tag = "[Primary]" if r["seq_num"] == 1 else ""
            dx_lines.append(f"  {int(r['seq_num'])}. {r['long_title']} {tag}".strip())
        dx_str = "\n".join(dx_lines)
    else:
        dx_str = "none listed"

    # --- Prescriptions (ground truth, hidden from model) ---
    meds = (rx_df[rx_df["hadm_id"] == hadm]["drug"]
            .dropna().astype(str).str.strip()
            .drop_duplicates().head(20).tolist())
    meds_str = ", ".join(meds) if meds else "none recorded"

    # --- Labs with human-readable names ---
    labs_slice = (labs_df[labs_df["hadm_id"] == hadm]
                  [["label", "valuenum", "valueuom", "flag"]]
                  .dropna(subset=["valuenum"]).head(15))
    if not labs_slice.empty:
        labs_lines = []
        for _, r in labs_slice.iterrows():
            flag = f" ({r['flag']})" if pd.notna(r.get("flag")) and str(r.get("flag", "")).strip() else ""
            labs_lines.append(f"  {str(r['label']):<30} {r['valuenum']:>8.2f}  {str(r['valueuom']):<10}{flag}")
        labs_str = "\n".join(labs_lines)
    else:
        labs_str = "no numeric labs available"

    # --- ICU stay ---
    icu_slice = icu_df[icu_df["hadm_id"] == hadm][["first_careunit", "los"]]
    icu_str = icu_slice.to_string(index=False) if not icu_slice.empty else "no ICU stay recorded"

    # --- ICU vitals (chartevents: HR, SpO2, Temp, BP, RR) ---
    if vitals_df is not None and not vitals_df.empty:
        vit_slice = (vitals_df[vitals_df["hadm_id"] == hadm]
                     [["label", "valuenum", "valueuom"]].head(12))
        vit_str = vit_slice.to_string(index=False) if not vit_slice.empty else "no ICU vitals recorded"
    else:
        vit_str = "no ICU vitals recorded"

    # --- OMR: BP, weight, BMI ---
    if omr_df is not None and not omr_df.empty:
        sid = row["subject_id"]
        omr_slice = omr_df[omr_df["subject_id"] == sid][["result_name", "result_value"]].head(8)
        omr_str = omr_slice.to_string(index=False) if not omr_slice.empty else "no OMR data"
    else:
        omr_str = "no OMR data"

    # --- Procedures ---
    if proc_df is not None and not proc_df.empty:
        proc_slice = proc_df[proc_df["hadm_id"] == hadm]["long_title"].dropna().head(10).tolist()
        proc_str = "\n  ".join(proc_slice) if proc_slice else "none recorded"
    else:
        proc_str = "none recorded"

    # --- Microbiology cultures ---
    if micro_df is not None and not micro_df.empty:
        mic_slice = (micro_df[micro_df["hadm_id"] == hadm]
                     [["spec_type_desc", "org_name", "ab_name", "interpretation"]].head(6))
        micro_str = mic_slice.to_string(index=False) if not mic_slice.empty else "no cultures recorded"
    else:
        micro_str = "no cultures recorded"

    prompt = f"""You are a senior attending physician reviewing a hospital admission record.
Analyze the data below and respond in the exact structured format requested.

== PATIENT DEMOGRAPHICS ==
{demo}

== DIAGNOSES (Primary + Comorbidities) ==
{dx_str}

== LAB RESULTS ==
{labs_str}

== ICU VITALS (Heart Rate / SpO2 / Temp / BP / RR) ==
{vit_str}

== VITAL SIGNS & ANTHROPOMETRICS (OMR) ==
{omr_str}

== PROCEDURES PERFORMED ==
  {proc_str}

== MICROBIOLOGY / CULTURES ==
{micro_str}

== ICU STAY ==
{icu_str}

== YOUR TASK ==
Provide your clinical analysis in EXACTLY this format:

(1) Additional Diagnosis Hypotheses
    - List 3 to 5 plausible differential or comorbid diagnoses.
    - For each, give a one-sentence clinical rationale.

(2) Recommended Treatments
    - Provide specific drug names (and typical doses where reasonable).
    - Include non-pharmacological recommendations if relevant.
    - Justify each briefly.

(3) Safety Flags / Drug Interactions
    - List any concerning interactions, contraindications, or monitoring needs.

(4) Confidence Level: Low | Medium | High
    - One sentence explaining your confidence.

Be concise but clinically precise. Avoid generic boilerplate."""
    return prompt, meds_str


# =====================================================================
# 4. API CALL WRAPPER (WITH MULTI-KEY ROTATION & 7 PROVIDERS)
# Model 1 -> Groq -> OpenRouter -> SambaNova -> Fireworks -> NVIDIA NIM
# Evaluator -> Groq -> OpenRouter -> Cerebras -> NVIDIA NIM
# Model 2 -> Google Gemini API
# =====================================================================
import time
import os
from colorama import Fore

def load_keys(filename):
    if not os.path.exists(filename): return []
    with open(filename) as f: return [line.strip() for line in f if line.strip()]

GROQ_KEYS = load_keys("groq_api_key.txt")
OPENROUTER_KEYS = load_keys("openrouter_api_key.txt")
GEMINI_KEYS = load_keys("gemini_api_key.txt")
CEREBRAS_KEYS = load_keys("cerebras_api_key.txt")
SAMBANOVA_KEYS = load_keys("sambanova_api_key.txt")
FIREWORKS_KEYS = load_keys("fireworks_api_key.txt")
NVIDIA_KEYS = load_keys("nvidia_api_key.txt")

for fname, keys in [
    ("groq_api_key.txt", GROQ_KEYS),
    ("openrouter_api_key.txt", OPENROUTER_KEYS),
    ("gemini_api_key.txt", GEMINI_KEYS),
    ("cerebras_api_key.txt", CEREBRAS_KEYS),
    ("sambanova_api_key.txt", SAMBANOVA_KEYS),
    ("fireworks_api_key.txt", FIREWORKS_KEYS),
    ("nvidia_api_key.txt", NVIDIA_KEYS),
]:
    if not keys:
        print(Fore.YELLOW + f"Warning: {fname} is missing or empty!")

try:
    from groq import Groq
except ImportError:
    raise SystemExit("Fatal Error: groq library not found. Run 'pip install groq'")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Fatal Error: openai library not found. Run 'pip install openai'")

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    raise SystemExit("Fatal Error: google.generativeai library not found. Run 'pip install google-generativeai'")

GROQ_KEY_IDX = 0
OPENROUTER_KEY_IDX = 0
GEMINI_KEY_IDX = 0
CEREBRAS_KEY_IDX = 0
SAMBANOVA_KEY_IDX = 0
FIREWORKS_KEY_IDX = 0
NVIDIA_KEY_IDX = 0

groq_client      = Groq(api_key=GROQ_KEYS[GROQ_KEY_IDX]) if GROQ_KEYS else None
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEYS[OPENROUTER_KEY_IDX]) if OPENROUTER_KEYS else None
cerebras_client  = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_KEYS[CEREBRAS_KEY_IDX]) if CEREBRAS_KEYS else None
sambanova_client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=SAMBANOVA_KEYS[SAMBANOVA_KEY_IDX]) if SAMBANOVA_KEYS else None
fireworks_client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FIREWORKS_KEYS[FIREWORKS_KEY_IDX]) if FIREWORKS_KEYS else None
nvidia_client    = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_KEYS[NVIDIA_KEY_IDX]) if NVIDIA_KEYS else None

if GEMINI_KEYS:
    genai.configure(api_key=GEMINI_KEYS[GEMINI_KEY_IDX])

# Model name mappings per provider
GROQ_TO_OPENROUTER = {
    "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b-instant":    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-oss-120b":     "openai/gpt-oss-120b",
}
GROQ_TO_SAMBANOVA = {
    "llama-3.3-70b-versatile": "Meta-Llama-3.3-70B-Instruct",
    "llama-3.1-8b-instant":    "Meta-Llama-3.1-8B-Instruct",
}
GROQ_TO_FIREWORKS = {
    "llama-3.3-70b-versatile": "accounts/fireworks/models/llama-v3p3-70b-instruct",
}
GROQ_TO_NVIDIA = {
    "llama-3.3-70b-versatile": "meta/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b":     "openai/gpt-oss-120b",
}
OPENROUTER_TO_CEREBRAS = {
    "openai/gpt-oss-120b": "llama3.1-70b",
}

# Fallback states:
# model1:    0=Groq, 1=OpenRouter, 2=SambaNova, 3=Fireworks, 4=NVIDIA
# evaluator: 0=Groq, 1=OpenRouter, 2=Cerebras,  3=NVIDIA
FALLBACK_STATE = {
    "model1": 0,
    "evaluator": 0,
}

def ask(model, prompt, max_retries=9999, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    global GROQ_KEY_IDX, OPENROUTER_KEY_IDX, GEMINI_KEY_IDX, CEREBRAS_KEY_IDX
    global SAMBANOVA_KEY_IDX, FIREWORKS_KEY_IDX, NVIDIA_KEY_IDX
    global groq_client, openrouter_client, cerebras_client
    global sambanova_client, fireworks_client, nvidia_client

    empty_attempts = 0

    for attempt in range(max_retries):
        try:
            is_m1   = (model == MODEL1_NAME)
            is_eval = (model == EVALUATOR_NAME)

            # ── MODEL 2: Google Gemini ─────────────────────────────────────
            if model == MODEL2_NAME:
                api_type = "gemini"
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                gemini_model = genai.GenerativeModel(model_name=model, generation_config=generation_config)
                resp = gemini_model.generate_content(prompt)
                out  = resp.text

            # ── MODEL 1: Groq -> OpenRouter -> SambaNova -> Fireworks -> NVIDIA
            elif is_m1:
                state = FALLBACK_STATE["model1"]
                if state == 0 and GROQ_KEYS:
                    api_type = "groq"
                    resp = groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                elif state == 1 and OPENROUTER_KEYS:
                    api_type = "openrouter"
                    resp = openrouter_client.chat.completions.create(
                        model=GROQ_TO_OPENROUTER.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                elif state == 2 and SAMBANOVA_KEYS:
                    api_type = "sambanova"
                    resp = sambanova_client.chat.completions.create(
                        model=GROQ_TO_SAMBANOVA.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                elif state == 3 and FIREWORKS_KEYS:
                    api_type = "fireworks"
                    resp = fireworks_client.chat.completions.create(
                        model=GROQ_TO_FIREWORKS.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                else:
                    api_type = "nvidia"
                    resp = nvidia_client.chat.completions.create(
                        model=GROQ_TO_NVIDIA.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content

            # ── EVALUATOR: Groq -> OpenRouter -> Cerebras -> NVIDIA ────────
            elif is_eval:
                state = FALLBACK_STATE["evaluator"]
                if state == 0 and GROQ_KEYS:
                    api_type = "groq"
                    resp = groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                elif state == 1 and OPENROUTER_KEYS:
                    api_type = "openrouter"
                    resp = openrouter_client.chat.completions.create(
                        model=GROQ_TO_OPENROUTER.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                elif state == 2 and CEREBRAS_KEYS:
                    api_type = "cerebras"
                    resp = cerebras_client.chat.completions.create(
                        model=OPENROUTER_TO_CEREBRAS.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content
                else:
                    api_type = "nvidia"
                    resp = nvidia_client.chat.completions.create(
                        model=GROQ_TO_NVIDIA.get(model, model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content

            else:
                return "ERROR: unknown model or fallback state"

            # ── Output length check ────────────────────────────────────────
            if out and len(str(out).strip()) >= 50:
                return out
            else:
                actual_len = len(str(out).strip()) if out else 0
                empty_attempts += 1
                if empty_attempts >= 3:
                    return "ERROR: output too short"
                print(Fore.RED + f"  Warning: Output too short ({actual_len} chars). Retrying in 5s...")
                time.sleep(5)
                continue

        except Exception as e:
            err = str(e).lower()

            # ── 404 / Model Not Found ──────────────────────────────────────
            if "404" in err or "not found" in err:
                if api_type == "groq":
                    print(Fore.YELLOW + f"  Groq: model not found. Falling back...")
                    if is_m1:   FALLBACK_STATE["model1"]    = 1
                    if is_eval: FALLBACK_STATE["evaluator"] = 1
                    continue
                elif api_type == "openrouter":
                    print(Fore.YELLOW + f"  OpenRouter: model not found. Falling back...")
                    if is_m1:   FALLBACK_STATE["model1"]    = 2
                    if is_eval: FALLBACK_STATE["evaluator"] = 2
                    continue
                elif api_type == "sambanova" and is_m1:
                    print(Fore.YELLOW + f"  SambaNova: model not found. Falling back to Fireworks...")
                    FALLBACK_STATE["model1"] = 3
                    continue
                elif api_type == "fireworks" and is_m1:
                    print(Fore.YELLOW + f"  Fireworks: model not found. Falling back to NVIDIA...")
                    FALLBACK_STATE["model1"] = 4
                    continue
                elif api_type == "cerebras" and is_eval:
                    print(Fore.YELLOW + f"  Cerebras: model not found. Falling back to NVIDIA...")
                    FALLBACK_STATE["evaluator"] = 3
                    continue

            # ── Rate Limit / Insufficient Credits ─────────────────────────
            if "rate limit" in err or "429" in err or "resourceexhausted" in err or "402" in err or "credits" in err:

                if api_type == "gemini":
                    GEMINI_KEY_IDX += 1
                    if GEMINI_KEY_IDX < len(GEMINI_KEYS):
                        print(Fore.YELLOW + f"  Gemini limit. Switching to key {GEMINI_KEY_IDX+1}/{len(GEMINI_KEYS)}...")
                        genai.configure(api_key=GEMINI_KEYS[GEMINI_KEY_IDX])
                        continue
                    else:
                        raise SystemExit(f"FATAL: All {len(GEMINI_KEYS)} Gemini keys exhausted.")

                elif api_type == "groq":
                    GROQ_KEY_IDX += 1
                    if GROQ_KEY_IDX < len(GROQ_KEYS):
                        print(Fore.YELLOW + f"  Groq limit. Switching to key {GROQ_KEY_IDX+1}/{len(GROQ_KEYS)}...")
                        groq_client = Groq(api_key=GROQ_KEYS[GROQ_KEY_IDX])
                        continue
                    else:
                        print(Fore.YELLOW + "  All Groq keys exhausted! Falling back to OpenRouter...")
                        if is_m1:   FALLBACK_STATE["model1"]    = 1
                        if is_eval: FALLBACK_STATE["evaluator"] = 1
                        continue

                elif api_type == "openrouter":
                    OPENROUTER_KEY_IDX += 1
                    if OPENROUTER_KEY_IDX < len(OPENROUTER_KEYS):
                        print(Fore.YELLOW + f"  OpenRouter limit. Switching to key {OPENROUTER_KEY_IDX+1}/{len(OPENROUTER_KEYS)}...")
                        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEYS[OPENROUTER_KEY_IDX])
                        continue
                    else:
                        if is_m1:
                            print(Fore.YELLOW + "  All OpenRouter keys exhausted! Falling back to SambaNova...")
                            FALLBACK_STATE["model1"] = 2
                        elif is_eval:
                            print(Fore.YELLOW + "  All OpenRouter keys exhausted! Falling back to Cerebras...")
                            FALLBACK_STATE["evaluator"] = 2
                        continue

                elif api_type == "sambanova":
                    SAMBANOVA_KEY_IDX += 1
                    if SAMBANOVA_KEY_IDX < len(SAMBANOVA_KEYS):
                        print(Fore.YELLOW + f"  SambaNova limit. Switching to key {SAMBANOVA_KEY_IDX+1}/{len(SAMBANOVA_KEYS)}...")
                        sambanova_client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=SAMBANOVA_KEYS[SAMBANOVA_KEY_IDX])
                        continue
                    else:
                        print(Fore.YELLOW + "  All SambaNova keys exhausted! Falling back to Fireworks...")
                        FALLBACK_STATE["model1"] = 3
                        continue

                elif api_type == "fireworks":
                    FIREWORKS_KEY_IDX += 1
                    if FIREWORKS_KEY_IDX < len(FIREWORKS_KEYS):
                        print(Fore.YELLOW + f"  Fireworks limit. Switching to key {FIREWORKS_KEY_IDX+1}/{len(FIREWORKS_KEYS)}...")
                        fireworks_client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FIREWORKS_KEYS[FIREWORKS_KEY_IDX])
                        continue
                    else:
                        print(Fore.YELLOW + "  All Fireworks keys exhausted! Falling back to NVIDIA...")
                        FALLBACK_STATE["model1"] = 4
                        continue

                elif api_type == "cerebras":
                    CEREBRAS_KEY_IDX += 1
                    if CEREBRAS_KEY_IDX < len(CEREBRAS_KEYS):
                        print(Fore.YELLOW + f"  Cerebras limit. Switching to key {CEREBRAS_KEY_IDX+1}/{len(CEREBRAS_KEYS)}...")
                        cerebras_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_KEYS[CEREBRAS_KEY_IDX])
                        continue
                    else:
                        print(Fore.YELLOW + "  All Cerebras keys exhausted! Falling back to NVIDIA...")
                        FALLBACK_STATE["evaluator"] = 3
                        continue

                elif api_type == "nvidia":
                    NVIDIA_KEY_IDX += 1
                    if NVIDIA_KEY_IDX < len(NVIDIA_KEYS):
                        print(Fore.YELLOW + f"  NVIDIA limit. Switching to key {NVIDIA_KEY_IDX+1}/{len(NVIDIA_KEYS)}...")
                        nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_KEYS[NVIDIA_KEY_IDX])
                        continue
                    else:
                        raise SystemExit(f"FATAL: All {len(NVIDIA_KEYS)} NVIDIA keys exhausted. Add more keys and restart.")

            print(Fore.RED + f"  API Error ({type(e).__name__}): {e}. Retrying in 10s...")
            time.sleep(10)

    return "ERROR: failed"

# Regex to extract drug-like tokens from model output (used in drug overlap matching)
DRUG_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

def _normalize(s):
    """Lowercase + strip non-alpha characters for fuzzy drug name comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower().strip())

def find_drug_overlap(model_output, true_rx_df, threshold=FUZZY_THRESHOLD):

    """Fuzzy match drugs mentioned in model output against true prescriptions."""
    if not model_output or model_output.startswith("ERROR"):
        return 0, []

    true_drugs = {
        _normalize(d)
        for d in true_rx_df["drug"].dropna().astype(str).tolist()
        if len(d.strip()) >= 4
    }
    candidate_tokens = {t.lower() for t in DRUG_TOKEN_RE.findall(model_output)}

    matched = set()
    for cand in candidate_tokens:
        for true in true_drugs:
            ratio = difflib.SequenceMatcher(None, cand, true).ratio()
            if ratio >= threshold:
                matched.add(true)
                break
    return len(matched), sorted(matched)


# ===================== METRICS =====================================
def compute_metrics(results_df, eval_df=None):
    """Compute Precision@k, Recall, avg response length, avg eval score per model."""
    summary = {}
    model_map = {
        "model1": f"Model 1 ({MODEL1_NAME})",
        "model2": f"Model 2 ({MODEL2_NAME})",
    }
    for prefix, label in model_map.items():
        overlap_col = f"{prefix}_overlap"
        len_col     = f"{prefix}_len"
        true_col    = "true_rx_count"

        if overlap_col not in results_df.columns:
            continue

        overlaps = results_df[overlap_col].fillna(0)
        true_n   = results_df[true_col].fillna(1).clip(lower=1)

        precision_k = (overlaps.clip(upper=PRECISION_K) / PRECISION_K).mean()
        recall      = (overlaps / true_n).mean()
        avg_len     = results_df[len_col].mean()

        avg_eval = None
        if eval_df is not None and not eval_df.empty:
            eval_col = f"{prefix}_total"
            if eval_col in eval_df.columns:
                avg_eval = eval_df[eval_col].dropna().mean()

        summary[label] = {
            f"precision@{PRECISION_K}": round(precision_k, 3),
            "recall":                   round(recall, 3),
            "avg_response_length":      round(avg_len, 1),
            "avg_eval_score":           round(avg_eval, 2) if avg_eval is not None else "n/a",
        }
    return summary

# ========= EVALUATOR PROMPT, covers Model 1 AND 2 ==========================
EVALUATOR_PROMPT_TEMPLATE = """\
You are an impartial senior physician acting as an independent evaluator.
Two AI assistants analyzed the SAME patient record and produced clinical recommendations.


== PATIENT CONTEXT ==
{context}


== MODEL 1 OUTPUT ({m1_name}) ==
{m1}


== MODEL 2 OUTPUT ({m2_name}) ==
{m2}


== EVALUATION TASK ==
Score each model on 4 axes (1–25 each, total /100):
  - Clinical Accuracy    (1-25): correctness of differentials, drugs, doses
  - Completeness         (1-25): coverage of all relevant clinical aspects
  - Safety               (1-25): handling of interactions, contraindications, red flags
  - Practical Usefulness (1-25): clarity and actionability for a clinician


Respond with ONLY valid JSON — no markdown fences, no prose outside the JSON:
{{
  "model1": {{"accuracy": <int>, "completeness": <int>, "safety": <int>, "usefulness": <int>, "total": <int>, "reasoning": "<2-3 sentences>"}},
  "model2": {{"accuracy": <int>, "completeness": <int>, "safety": <int>, "usefulness": <int>, "total": <int>, "reasoning": "<2-3 sentences>"}},
  "winner": "model1" | "model2" | "tie",
  "comparison": "<1-2 sentences explaining the winner>"
}}"""


def parse_eval_json(text):
    """Extract JSON from evaluator response. Returns fallback dict on failure."""
    fallback = {
        "model1": {"accuracy": 0, "completeness": 0, "safety": 0,
                   "usefulness": 0, "total": 0, "reasoning": "parse_failed"},
        "model2": {"accuracy": 0, "completeness": 0, "safety": 0,
                   "usefulness": 0, "total": 0, "reasoning": "parse_failed"},
        "winner": "tie",
        "comparison": "evaluator output could not be parsed",
    }
    if not text or text.startswith("ERROR"):
        return fallback
    try:
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            return fallback
        return json.loads(text[start:end + 1])
    except Exception as e:
        print(Fore.YELLOW + f"  [parse_eval_json] {e}")
        return fallback


# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# This is where everything comes together. We load the data, loop through
# each patient, ask the models, run the evaluator, and save results.
# =====================================================================
def main():
    # ================================================================
    # PHASE 1: Get the full patient pool (lightweight — no heavy data)
    # ================================================================
    pool = get_patient_pool(MIMIC_DB, MAX_PATIENTS)
    if pool.empty:
        print(Fore.RED + "No eligible patients found. Exiting.")
        return
    print_cohort_summary(pool)

    # One row per patient for the main loop; pool keeps all dx rows for build_prompt
    unique_pool = pool.drop_duplicates(subset="hadm_id").reset_index(drop=True)
    total = len(unique_pool)

    # ================================================================
    # PHASE 2: Resume support — read only the hadm_id column to save RAM
    # ================================================================
    done_ids      = set()
    eval_done_ids = set()

    if os.path.exists(OUTPUT_CSV):
        done_ids = set(pd.read_csv(OUTPUT_CSV, usecols=["hadm_id"])["hadm_id"].tolist())
        print(Fore.YELLOW + f"[resume] {len(done_ids)} model results already saved")

    if os.path.exists(EVAL_CSV):
        eval_done_ids = set(pd.read_csv(EVAL_CSV, usecols=["hadm_id"])["hadm_id"].tolist())
        print(Fore.YELLOW + f"[resume] {len(eval_done_ids)} evaluations already saved\n")

    # ================================================================
    # PHASE 3: Batch loop — load heavy data BATCH_SIZE patients at a time
    # ================================================================
    for batch_start in range(0, total, BATCH_SIZE):
        batch_unique = unique_pool.iloc[batch_start : batch_start + BATCH_SIZE]

        # Skip entire batch if every patient in it is fully done
        batch_hids = batch_unique["hadm_id"].tolist()
        if all(
            hid in done_ids and (not RUN_EVALUATOR or hid in eval_done_ids)
            for hid in batch_hids
        ):
            print(Fore.YELLOW + f"[batch {batch_start+1}-{batch_start+len(batch_unique)}/{total}] all done, skipping")
            continue

        # Get all diagnosis rows for this batch (for build_prompt)
        batch_pool = pool[pool["hadm_id"].isin(batch_hids)]

        print(Fore.CYAN + f"\n[batch {batch_start+1}-{batch_start+len(batch_unique)}/{total}] loading clinical data...")
        labs_df, rx_df, icu_df, vitals_df, omr_df, proc_df, micro_df = get_batch_data(MIMIC_DB, batch_pool)

        # --- Process each patient in this batch ---
        for idx, row in batch_unique.iterrows():
            n   = idx + 1
            hid = row["hadm_id"]
            sid = row["subject_id"]

            is_eval_eligible = RUN_EVALUATOR and (EVAL_ONLY_FIRST_N is None or n <= EVAL_ONLY_FIRST_N)

            if hid in done_ids and (not is_eval_eligible or hid in eval_done_ids):
                print(Fore.YELLOW + f"  [{n}/{total}] subject {sid} already fully done, skipping")
                continue

            prompt, meds_str = build_prompt(row, batch_pool, labs_df, rx_df, icu_df, vitals_df, omr_df, proc_df, micro_df)
            true_rx       = rx_df[rx_df["hadm_id"] == hid]
            true_rx_count = true_rx["drug"].nunique()

            m1, m2 = "", ""
            if hid not in done_ids:
                print(Fore.CYAN + Style.BRIGHT +
                      f"\n  [{n}/{total}] subject={sid}  hadm={hid}  "
                      f"age={row['anchor_age']}  gender={row['gender']}")

                print(Fore.YELLOW + f"    -> {MODEL1_NAME}")
                m1 = ask(MODEL1_NAME, prompt)
                time.sleep(SLEEP_BETWEEN)

                print(Fore.YELLOW + f"    -> {MODEL2_NAME}")
                m2 = ask(MODEL2_NAME, prompt)
                time.sleep(SLEEP_BETWEEN)

                m1_ov, m1_matched = find_drug_overlap(m1, true_rx)
                m2_ov, m2_matched = find_drug_overlap(m2, true_rx)

                new_row = {
                    "patient_num":       n,
                    "subject_id":        sid,
                    "hadm_id":           hid,
                    "age":               row["anchor_age"],
                    "gender":            "male" if row["gender"] == "M" else "female",
                    "primary_dx":        row.get("long_title"),
                    "true_rx_count":     true_rx_count,
                    "prompt":            prompt,
                    "model1_output":     m1,
                    "model1_len":        len(m1) if m1 else 0,
                    "model1_overlap":    m1_ov,
                    "model1_rx_matched": "; ".join(m1_matched),
                    "model2_output":     m2,
                    "model2_len":        len(m2) if m2 else 0,
                    "model2_overlap":    m2_ov,
                    "model2_rx_matched": "; ".join(m2_matched),
                }
                # Append single row to CSV — never keep all rows in RAM
                write_header = not os.path.exists(OUTPUT_CSV)
                pd.DataFrame([new_row]).to_csv(
                    OUTPUT_CSV, mode="a", index=False,
                    header=write_header, encoding="utf-8-sig"
                )
                print(Fore.GREEN + f"    saved -> {OUTPUT_CSV}")
                done_ids.add(hid)
            else:
                # Models done but eval not done — read from CSV instead of RAM
                try:
                    row_data = pd.read_csv(OUTPUT_CSV, usecols=["hadm_id", "model1_output", "model2_output", "prompt"])
                    match = row_data[row_data["hadm_id"] == hid].iloc[0]
                    m1     = match["model1_output"]
                    m2     = match["model2_output"]
                    prompt = match["prompt"]
                except Exception:
                    print(Fore.RED + f"    Could not reload outputs for hadm {hid}, skipping eval.")
                    continue

            # --- Evaluator ---
            if is_eval_eligible and hid not in eval_done_ids:
                evaluator_context = f"{prompt}\n\n== GROUND TRUTH / ACTUAL PRESCRIBED MEDICATIONS ==\n{meds_str}"
                eval_prompt = EVALUATOR_PROMPT_TEMPLATE.format(
                    context=evaluator_context,
                    m1_name=MODEL1_NAME, m1=m1,
                    m2_name=MODEL2_NAME, m2=m2,
                )
                print(Fore.YELLOW + f"    -> evaluator ({EVALUATOR_NAME})")
                ev_raw = ask(EVALUATOR_NAME, eval_prompt, temperature=0.1, max_tokens=3000)
                time.sleep(SLEEP_BETWEEN)

                ev = parse_eval_json(ev_raw)

                eval_row = {
                    "patient_num":         n,
                    "subject_id":          sid,
                    "hadm_id":             hid,
                    "model1_accuracy":     ev["model1"]["accuracy"],
                    "model1_completeness": ev["model1"]["completeness"],
                    "model1_safety":       ev["model1"]["safety"],
                    "model1_usefulness":   ev["model1"]["usefulness"],
                    "model1_total":        ev["model1"]["total"],
                    "model1_reasoning":    ev["model1"]["reasoning"],
                    "model2_accuracy":     ev["model2"]["accuracy"],
                    "model2_completeness": ev["model2"]["completeness"],
                    "model2_safety":       ev["model2"]["safety"],
                    "model2_usefulness":   ev["model2"]["usefulness"],
                    "model2_total":        ev["model2"]["total"],
                    "model2_reasoning":    ev["model2"]["reasoning"],
                    "winner":              ev["winner"],
                    "comparison":          ev["comparison"],
                    "raw_eval":            ev_raw,
                }
                write_header = not os.path.exists(EVAL_CSV)
                pd.DataFrame([eval_row]).to_csv(
                    EVAL_CSV, mode="a", index=False,
                    header=write_header, encoding="utf-8-sig"
                )
                print(Fore.GREEN + f"    eval saved -> {EVAL_CSV}")
                eval_done_ids.add(hid)

            if n < total:
                print(Fore.YELLOW + f"    waiting {SLEEP_BETWEEN_PAT}s before next patient...")
                time.sleep(SLEEP_BETWEEN_PAT)

        # Free batch memory before loading the next batch
        del labs_df, rx_df, icu_df, vitals_df, omr_df, proc_df, micro_df

    # ================================================================
    # PHASE 4: Final metrics (read from CSV, never from RAM)
    # ================================================================
    results_df = pd.read_csv(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else pd.DataFrame()
    eval_df    = pd.read_csv(EVAL_CSV)   if os.path.exists(EVAL_CSV)   else None

    print(Fore.GREEN + Style.BRIGHT + f"\n=== DONE — {len(results_df)} patients processed ===")
    print(Fore.GREEN + f"Results -> {OUTPUT_CSV}")
    if eval_df is not None:
        print(Fore.GREEN + f"Evals   -> {EVAL_CSV} ({len(eval_df)} patients scored)")

    metrics = compute_metrics(results_df, eval_df)
    print(Fore.CYAN + "\n=== METRICS ===")
    for model_label, m in metrics.items():
        print(Fore.CYAN + f"  {model_label}")
        for k, v in m.items():
            print(f"    {k:<25s}: {v}")


if __name__ == "__main__":
    main()
