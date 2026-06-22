# ============================================================
# BSP2 / Project Omega -- Step 3: Temperature Experiment
#
# RUN ORDER:
# 1. patients_pipeline.py    (Generates LLM responses & evaluator scores)
# 2. bleu_rouge_metrics.py      (Calculates text similarity metrics)
# 3. temperature_test.py             <-- THIS FILE (Runs the hyperparameter experiment)
# 4. graph_maker.py                  (Builds the main visualization dashboard)
# 5. lime_analyzer.py           (Runs LIME explainability analysis)
# ============================================================

import os
import time
import pandas as pd
try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("Fatal Error: google.generativeai not found. Run 'pip install google-generativeai'")
from colorama import init, Fore
init(autoreset=True)

# =====================================================================
# 1. API SETUP
# Model: gemma-4-31b-it via Google Gemini API
# Multi-key rotation: add multiple keys to gemini_api_key.txt
# =====================================================================
def load_keys(filename):
    if not os.path.exists(filename): return []
    with open(filename) as f: return [line.strip() for line in f if line.strip()]

GEMINI_KEYS = load_keys("gemini_api_key.txt")

if not GEMINI_KEYS:
    raise SystemExit("Fatal Error: gemini_api_key.txt is missing or empty!")

GEMINI_KEY_IDX = 0
genai.configure(api_key=GEMINI_KEYS[GEMINI_KEY_IDX])

# =====================================================================
# 2. CONFIGURATION
# =====================================================================
MODEL_NAME    = "gemma-4-31b-it"   # Runs via Google Gemini API
TEMPERATURES  = [0.1, 0.5, 1.0]   # Low / Balanced / Creative
TEST_PATIENTS = 1000     # Number of patients to use
SLEEP         = 3       # Seconds between API calls
MAX_TOKENS    = 2500    # Match main pipeline token limit
MIN_LEN       = 200     

# =====================================================================
# 3. API CALL WRAPPER — Google Gemini with multi-key rotation
# =====================================================================
def ask_temp(model, content, temperature, max_retries=10):
    global GEMINI_KEY_IDX

    for attempt in range(max_retries):
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": MAX_TOKENS,
            }
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
            )
            resp = gemini_model.generate_content(content)
            return resp.text

        except Exception as e:
            err = str(e).lower()

            if "rate limit" in err or "429" in err or "resourceexhausted" in err or "quota" in err:
                GEMINI_KEY_IDX += 1
                if GEMINI_KEY_IDX < len(GEMINI_KEYS):
                    print(Fore.YELLOW + f"    Gemini limit. Switching to key {GEMINI_KEY_IDX+1}/{len(GEMINI_KEYS)}...")
                    genai.configure(api_key=GEMINI_KEYS[GEMINI_KEY_IDX])
                    continue
                else:
                    raise SystemExit(f"Fatal: All {len(GEMINI_KEYS)} Gemini keys are rate-limited. Add more keys and restart.")

            print(Fore.RED + f"    API Error ({type(e).__name__}): {e}. Retrying in 10s...")
            time.sleep(10)

    raise SystemExit(f"Fatal: Failed after {max_retries} attempts. Exiting.")



# =====================================================================
# 4. MAIN PIPELINE
# Loops through the specified patients and generates outputs for each 
# temperature setting, saving incrementally to prevent data loss.
# =====================================================================
def main():
    if not os.path.exists("mimic_batch_results.csv"):
        raise FileNotFoundError("mimic_batch_results.csv not found!")

    # Fetch only the N patients designated for the experiment from the main batch
    df = pd.read_csv("mimic_batch_results.csv")
    if TEST_PATIENTS:
        df = df.head(TEST_PATIENTS)

    records = []
    done_keys = set()
    if os.path.exists("temperature_results.csv"):
        existing = pd.read_csv("temperature_results.csv")
        records = existing.to_dict("records")
        for r in records:
            done_keys.add(f"{r['hadm_id']}_{r['temperature']}")
        print(Fore.YELLOW + f"[resume] {len(done_keys)} temperature experiments already done")

    print(Fore.MAGENTA + f"=== TEMPERATURE EXPERIMENT STARTING ({len(df)} Patients) ===")
    
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if pd.isna(prompt):
            continue
            
        for temp in TEMPERATURES:
            if f"{row['hadm_id']}_{temp}" in done_keys:
                print(Fore.YELLOW + f"  Patient {row['subject_id']} (HADM: {row['hadm_id']}) | temp={temp} -> Already done, skipping")
                continue

            print(Fore.CYAN + f"  Patient {row['subject_id']} (HADM: {row['hadm_id']}) | temp={temp}")
            out = None
            for attempt in range(3):
                out = ask_temp(MODEL_NAME, prompt, temp)
                if out and len(str(out).strip()) >= MIN_LEN:
                    break
                actual_len = len(str(out).strip()) if out else 0
                print(Fore.RED + f"    Warning: Output too short ({actual_len} chars, min={MIN_LEN}). (Attempt {attempt+1}/3). Retrying...")
                time.sleep(5)
            
            if not out or len(str(out).strip()) < MIN_LEN:
                raise SystemExit(f"Fatal Error: Output still too short after 3 attempts. Exiting immediately to prevent saving bad data.")
                
            records.append({
                "subject_id":    row["subject_id"],
                "hadm_id":       row["hadm_id"],
                "temperature":   temp,
                "output":        out,
                "output_length": len(out)
            })
            
            # Save to CSV after every call to prevent data loss in case of interruption
            pd.DataFrame(records).to_csv("temperature_results.csv", index=False, encoding="utf-8-sig")
            
            time.sleep(SLEEP)

    print(Fore.GREEN + f"\nDone! {len(records)} results successfully saved to 'temperature_results.csv'.")

if __name__ == "__main__":
    main()
