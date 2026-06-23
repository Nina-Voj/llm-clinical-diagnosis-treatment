# ============================================================
# BSP2 / Project Omega -- Step 5: LIME Explainability Analysis
# 
# RUN ORDER:
# 1. patients_pipeline.py       (Generates LLM responses & evaluator scores)
# 2. bleu_rouge_metrics.py      (Calculates text similarity metrics)
# 3. temperature_test.py        (Runs the hyperparameter experiment)
# 4. graph_maker.py             (Builds the main visualization dashboard)
# 5. lime_analyzer.py           <-- THIS FILE (Runs LIME explainability analysis)
#
# What it does:
#   Applies LIME (Local Interpretable Model-agnostic Explanations)
#   to explain which words in the clinical prompt most influenced
#   each LLM's output, using TF-IDF cosine similarity as a proxy
#   scoring function (no extra API calls needed).
#
# Install:  pip install lime scikit-learn matplotlib numpy pandas colorama
# Python:   3.10+
# ============================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_text import LimeTextExplainer

from colorama import init, Fore, Style
init(autoreset=True)

# =====================================================================
# 1. CONFIGURATION
# We define which files to read/write, how many patients to analyze, 
# and how many perturbation samples LIME should use (more samples = better quality).
# =====================================================================
INPUT_CSV   = "mimic_batch_results.csv"
OUTPUT_CSV  = "lime_results.csv"
CHART_FILE  = "lime_explainability_chart.png"

N_PATIENTS  = None  # Set to a number (eg. 5) to limit, or None to process all patients
N_FEATURES  = 10    # top words per LIME explanation
N_SAMPLES   = 800   # LIME perturbation samples (more = stable, slower)

MODEL_COLS = {
    "llama-3.3-70b-versatile": "model1_output",  # Llama 3.3 70B (Groq)
    "gemma-4-31b-it":          "model2_output",  # Gemma 4 31B (Google Gemini API)
}

# Prompt section headers (used for section-level aggregation)
PROMPT_SECTIONS = {
    "Demographics":  r"== PATIENT DEMOGRAPHICS ==",
    "Diagnoses":     r"== DIAGNOSES \(Primary \+ Comorbidities\) ==",
    "Lab Results":   r"== LAB RESULTS ==",
    "Vitals/OMR":    r"== (?:ICU VITALS|VITAL SIGNS & ANTHROPOMETRICS)",
    "Procedures":    r"== PROCEDURES PERFORMED ==",
    "Microbiology":  r"== MICROBIOLOGY / CULTURES ==",
    "ICU Stay":      r"== ICU STAY ==",
    "Task":          r"== YOUR TASK ==",
}

# =====================================================================
# 2. SCORING FUNCTION (TF-IDF SIMILARITY)
# LIME needs to know how much a word affects the model's output. 
# We measure this by removing words and checking how much the new text 
# differs from the original LLM output using TF-IDF cosine similarity.
# =====================================================================
def make_predict_fn(original_output: str):
    """
    Returns a LIME-compatible predict_fn.

    For each perturbed prompt, we measure TF-IDF cosine similarity
    between the perturbed text and the original LLM output.

    Interpretation: words whose removal reduces similarity to the
    original output are deemed 'influential' (positive LIME weight).
    Words whose removal increases similarity get negative weight.

    No additional API calls are required.
    """
    vectorizer = TfidfVectorizer(
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",  # alphabetic tokens only
    )
    try:
        ref_vec = vectorizer.fit_transform([original_output])
    except Exception:
        # fallback: return a flat scorer
        def flat_fn(texts):
            return np.tile([0.5, 0.5], (len(texts), 1))
        return flat_fn

    def predict_fn(texts: list[str]) -> np.ndarray:
        scores = np.zeros((len(texts), 2))
        for i, text in enumerate(texts):
            if not text or not text.strip():
                sim = 0.0
            else:
                try:
                    vec = vectorizer.transform([text])
                    sim = float(cosine_similarity(vec, ref_vec)[0][0])
                except Exception:
                    sim = 0.0
            sim = float(np.clip(sim, 0.0, 1.0))
            scores[i] = [1.0 - sim, sim]   # [not_influential, influential]
        return scores

    return predict_fn


# =====================================================================
# SECTION MAPPER
# =====================================================================
def word_to_section(word: str, prompt: str) -> str:
    """
    Find which prompt section a word belongs to by scanning the prompt
    text for section headers and the word's position.
    """
    prompt_lower  = prompt.lower()
    word_lower    = word.lower()
    word_pos      = prompt_lower.find(word_lower)
    if word_pos == -1:
        return "Other"

    # Build section start positions
    section_starts = {}
    for sec_name, pattern in PROMPT_SECTIONS.items():
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            section_starts[sec_name] = m.start()

    if not section_starts:
        return "Other"

    # Find the last section header that comes BEFORE the word
    best_sec   = "Other"
    best_start = -1
    for sec_name, start in section_starts.items():
        if start <= word_pos and start > best_start:
            best_start = start
            best_sec   = sec_name
    return best_sec


# =====================================================================
# 3. LIME RUNNER
# This function actually runs the LIME explainer on a single patient's prompt
# and returns a list of the most influential words and their weights.
# =====================================================================
def run_lime(prompt: str, output: str, model_label: str) -> list[dict]:
    """Run LIME on one (prompt, output) pair. Returns feature list."""
    explainer = LimeTextExplainer(
        class_names=["Not Influential", "Influential"],
        split_expression=r"\s+",
        bow=True,
        random_state=42,
    )
    predict_fn = make_predict_fn(output)
    try:
        exp = explainer.explain_instance(
            prompt,
            predict_fn,
            num_features=N_FEATURES,
            num_samples=N_SAMPLES,
            labels=[1],
        )
        return [
            {"word": w, "weight": float(v), "model": model_label}
            for w, v in exp.as_list(label=1)
        ]
    except Exception as exc:
        print(Fore.RED + f"      X LIME error ({model_label}): {exc}")
        return []


# =====================================================================
# VISUALIZATION
# =====================================================================
BG_DARK    = "#0d0f18"
BG_PANEL   = "#161926"
BG_PANEL2  = "#1c2030"
COL_POS    = "#4ecdc4"   # teal  -- positive influence
COL_NEG    = "#e05c5c"   # coral -- negative influence
COL_TEXT   = "#e8eaf0"
COL_MUTED  = "#8890a8"

MODEL_COLORS = {
    "llama-3.3-70b-versatile": "#4c72b0",
    "gemma-4-31b-it":  "#dd8452",
}


def plot_dashboard(all_results: list[dict], output_path: str):
    """
    Two-part figure:
      Top:    1 × 2 grid of global aggregated horizontal bar charts
      Bottom: Section-level importance heatmap (which prompt section
              drives each model most on average)
    """
    df = pd.DataFrame(all_results)
    total_patients = df["patient_label"].nunique() if not df.empty else 0
    models = list(MODEL_COLS.keys())
    n_mod = len(models)

    # Wider figure for readability
    fig = plt.figure(figsize=(40, 22), facecolor=BG_DARK)

    # ---- outer grid: top (bar charts) + bottom (heatmap) ----
    outer_gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[12.0, 6.0],
        hspace=0.20,
    )

    # ---- TOP: bar-chart grid ----
    inner_gs = GridSpecFromSubplotSpec(
        1, n_mod,
        subplot_spec=outer_gs[0],
        wspace=0.30,
    )

    fig.text(
        0.5, 0.985,
        f"Global LIME Explainability -- Top Influential Words (Aggregated across {total_patients} Patients)",
        ha="center", va="top", fontsize=40, fontweight="bold",
        color=COL_TEXT, fontfamily="monospace",
    )
    fig.text(
        0.5, 0.954,
        f"BSP2 / Project Omega  |  TF-IDF lexical-overlap proxy  |  n_samples={N_SAMPLES}",
        ha="center", va="top", fontsize=24, color=COL_MUTED,
    )

    for mi, model_name in enumerate(models):
        ax = fig.add_subplot(inner_gs[0, mi])
        ax.set_facecolor(BG_PANEL)

        sub = df[df["model"] == model_name]

        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=COL_MUTED, transform=ax.transAxes, fontsize=8)
            ax.axis("off")
        else:
            # Aggregate globally
            agg = sub.groupby("word").agg(
                mean_abs_weight=("weight", lambda x: np.mean(np.abs(x))),
                mean_raw_weight=("weight", "mean")
            ).reset_index()
            
            # Top 20 words by absolute influence
            agg = agg.sort_values("mean_abs_weight", ascending=True).tail(20)
            
            words   = agg["word"].tolist()
            weights = agg["mean_raw_weight"].tolist()
            colors  = [COL_POS if w >= 0 else COL_NEG for w in weights]

            ax.barh(words, weights, color=colors,
                    edgecolor="none", height=0.70)
            ax.axvline(0, color="#3a3f55", linewidth=1.2, linestyle="--")

            # value annotations
            for bar_w, word in zip(weights, words):
                ax.text(
                    bar_w + (0.001 if bar_w >= 0 else -0.001),
                    word,
                    f"{bar_w:+.3f}",
                    va="center",
                    ha="left" if bar_w >= 0 else "right",
                    fontsize=16, color=COL_MUTED,
                )

        # Add padding to prevent text from spilling over
        ax.margins(x=0.15)

        # Styling
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=COL_MUTED, labelsize=18, length=0)
        ax.set_xlabel("Mean LIME Weight (across all patients)", fontsize=18, color=COL_MUTED, labelpad=3)

        # Column header: model name
        color = MODEL_COLORS.get(model_name, COL_TEXT)
        ax.set_title(
            model_name,
            color=color, fontsize=28, fontweight="bold", pad=12,
        )

    # ---- BOTTOM: section-level heatmap ----
    ax_heat = fig.add_subplot(outer_gs[1])
    ax_heat.set_facecolor(BG_PANEL2)

    # Aggregate: mean |weight| per model x section
    sections = list(PROMPT_SECTIONS.keys()) + ["Other"]
    heat_data = np.zeros((n_mod, len(sections)))

    for mi, model_name in enumerate(models):
        for si, sec in enumerate(sections):
            vals = [abs(r["weight"]) for r in all_results
                    if r["model"] == model_name and r.get("section") == sec]
            heat_data[mi, si] = np.mean(vals) if vals else 0.0

    im = ax_heat.imshow(
        heat_data, aspect="auto", cmap="YlOrRd",
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(len(sections)))
    ax_heat.set_xticklabels(sections, color=COL_TEXT, fontsize=20)
    ax_heat.set_yticks(range(n_mod))
    ax_heat.set_yticklabels(models, color=COL_TEXT, fontsize=20)
    ax_heat.set_title(
        f"Section-Level Importance Heatmap  (aggregated across all {total_patients} patients)",
        color=COL_TEXT, fontsize=26, pad=16,
    )
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)

    # Cell value annotations
    for mi in range(n_mod):
        for si in range(len(sections)):
            v = heat_data[mi, si]
            # YlOrRd maps low values to light yellow and high values to dark red.
            # So low values should have dark text, high values should have light text.
            ax_heat.text(
                si, mi, f"{v:.3f}",
                ha="center", va="center",
                fontsize=20,
                color="black" if v <= heat_data.max() * 0.5 else COL_TEXT,
            )

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        pad=0.01, fraction=0.03)
    cbar.set_label("Mean |LIME Weight|", color=COL_MUTED, fontsize=18)
    cbar.ax.yaxis.set_tick_params(color=COL_MUTED, labelsize=18)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COL_MUTED)

    # ---- Legend ----
    pos_p = mpatches.Patch(color=COL_POS, label="Positive influence (word ↑ output similarity)")
    neg_p = mpatches.Patch(color=COL_NEG, label="Negative influence (word ↓ output similarity)")
    fig.legend(
        handles=[pos_p, neg_p],
        loc="lower center", ncol=2, fontsize=20,
        facecolor=BG_PANEL, labelcolor=COL_TEXT,
        framealpha=0.9, edgecolor="#2a2f45",
        bbox_to_anchor=(0.5, 0.002),
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(Fore.GREEN + f"  Chart saved -> {output_path}")


# =====================================================================
# 4. MAIN PIPELINE
# Loops through the models and patients, runs LIME, supports resuming 
# from where it left off, saves results, and generates the final dashboard.
# =====================================================================
def main():
    print(Fore.CYAN + Style.BRIGHT +
          "=" * 58)
    print(Fore.CYAN + Style.BRIGHT +
          "  BSP2 / Project Omega -- LIME Explainability Analysis")
    print(Fore.CYAN + Style.BRIGHT +
          "=" * 58 + "\n")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"'{INPUT_CSV}' not found.\n"
            "Make sure to run patients_pipeline.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    if N_PATIENTS:
        df = df.head(N_PATIENTS)
    print(Fore.YELLOW + f"  Loaded {len(df)} patients from '{INPUT_CSV}'")
    print(Fore.YELLOW + f"  Models : {', '.join(MODEL_COLS.keys())}")
    print(Fore.YELLOW + f"  LIME   : {N_FEATURES} features, {N_SAMPLES} samples\n")

    all_results = []
    csv_records = []
    done_models_per_pat = {}
    
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            csv_records = existing.to_dict("records")
            for r in csv_records:
                pat = str(r.get("subject_id"))
                mod = str(r.get("model"))
                if pat not in done_models_per_pat:
                    done_models_per_pat[pat] = set()
                done_models_per_pat[pat].add(mod)
                
                all_results.append({
                    "word": r.get("word"),
                    "weight": r.get("lime_weight"),
                    "model": mod,
                    "patient_label": f"P{r.get('patient_num')}: {str(r.get('primary_dx'))[:26]}…" if len(str(r.get('primary_dx'))) > 28 else f"P{r.get('patient_num')}: {str(r.get('primary_dx'))}",
                    "subject_id": pat,
                    "primary_dx": str(r.get("primary_dx")),
                    "section": r.get("prompt_section")
                })
            print(Fore.YELLOW + f"  [resume] Loaded {len(csv_records)} past LIME records.")
        except Exception as e:
            print(Fore.RED + f"  [resume] Could not load past LIME records: {e}")

    for idx, row in df.iterrows():
        pat_num    = int(idx) + 1
        subject_id = row.get("subject_id", "?")
        primary_dx = str(row.get("primary_dx", "Unknown"))
        prompt     = str(row.get("prompt", ""))

        # Short label for charts
        dx_short = primary_dx if len(primary_dx) <= 28 else primary_dx[:26] + "..."
        pat_label = f"P{pat_num}: {dx_short}"

        for model_name, col in MODEL_COLS.items():
            print(f"\rProcessing patient {pat_num}/{len(df)} [LIME: {model_name}]...", end="", flush=True)
            if str(subject_id) in done_models_per_pat and model_name in done_models_per_pat[str(subject_id)]:
                continue

            output = str(row.get(col, ""))
            if not output or output.startswith("ERROR") or len(output) < 10:
                continue

            features = run_lime(prompt, output, model_name)

            for feat in features:
                section = word_to_section(feat["word"], prompt)
                feat["patient_label"] = pat_label
                feat["subject_id"]    = subject_id
                feat["primary_dx"]    = primary_dx
                feat["section"]       = section
                all_results.append(feat)

                csv_records.append({
                    "patient_num":  pat_num,
                    "subject_id":   subject_id,
                    "primary_dx":   primary_dx,
                    "model":        model_name,
                    "word":         feat["word"],
                    "lime_weight":  round(feat["weight"], 6),
                    "prompt_section": section,
                    "influence":    "positive" if feat["weight"] >= 0 else "negative",
                })
                
            # Save after every model to prevent data loss
            pd.DataFrame(csv_records).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print()

    # ---- Final Save CSV ----
    results_df = pd.DataFrame(csv_records)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + f"  Results saved -> '{OUTPUT_CSV}'  ({len(csv_records)} rows)\n")

    # ---- Plot ----
    if all_results:
        print(Fore.CYAN + "  Building visualization ...")
        plot_dashboard(all_results, CHART_FILE)
    else:
        print(Fore.RED + "  No LIME results to visualize.")
        return

    # ---- Console summary ----
    print(Fore.MAGENTA + Style.BRIGHT + "\n" + "=" * 58)
    print(Fore.MAGENTA + Style.BRIGHT + "  TOP 5 INFLUENTIAL WORDS PER MODEL (avg |weight|)")
    print(Fore.MAGENTA + Style.BRIGHT + "=" * 58)
    for model_name in MODEL_COLS:
        subset = results_df[results_df["model"] == model_name]
        if subset.empty:
            continue
        top = (subset.groupby("word")["lime_weight"]
                     .apply(lambda x: x.abs().mean())
                     .sort_values(ascending=False)
                     .head(5))
        print(Fore.CYAN + f"\n  {model_name}")
        for word, weight in top.items():
            bar = "#" * int(weight * 200)
            print(f"    {word:<28} {weight:.4f}  {bar}")

    print(Fore.MAGENTA + Style.BRIGHT + "\n" + "=" * 58)
    print(Fore.MAGENTA + Style.BRIGHT + "  MOST INFLUENTIAL PROMPT SECTION PER MODEL")
    print(Fore.MAGENTA + Style.BRIGHT + "=" * 58)
    for model_name in MODEL_COLS:
        subset = results_df[results_df["model"] == model_name]
        if subset.empty:
            continue
        best_sec = (subset.groupby("prompt_section")["lime_weight"]
                          .apply(lambda x: x.abs().mean())
                          .idxmax())
        print(Fore.CYAN + f"  {model_name:<22} -> {best_sec}")

    print(Fore.GREEN + Style.BRIGHT + f"\n  OK Done!\n")
    print(f"    {OUTPUT_CSV:<38} -- word-level LIME weights")
    print(f"    {CHART_FILE:<38} -- visualization dashboard\n")


if __name__ == "__main__":
    main()
