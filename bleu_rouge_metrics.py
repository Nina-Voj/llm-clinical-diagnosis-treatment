# ============================================================
# BSP2 / Project Omega -- Step 2: BLEU & ROUGE Metrics
# 
# RUN ORDER:
# 1. patients_pipeline.py     (Generates LLM responses & evaluator scores)
# 2. bleu_rouge_metrics.py      <-- THIS FILE (Calculates text similarity metrics)
# 3. temperature_test.py        (Runs the hyperparameter experiment)
# 4. graph_maker.py             (Builds the main visualization dashboard)
# 5. lime_analyzer.py           (Runs LIME explainability analysis)
# ============================================================
import pandas as pd
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------- CONFIG -------------------
INPUT_CSV  = "mimic_batch_results.csv"
OUTPUT_CSV = "bleu_rouge_results.csv"

MODEL_COLS = {
    "llama-3.3-70b":  "model1_output",  # Llama 3.3 70B (Groq -> OpenRouter -> SambaNova -> Fireworks -> NVIDIA)
    "gemma-4-31b":    "model2_output",  # Gemma 4 31B (Google Gemini API)
}

# -------------- SCORER --------------------
scorer   = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smoother = SmoothingFunction().method1

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens  = nltk.word_tokenize(reference.lower())
    hyp_tokens  = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)

def compute_rouge(reference: str, hypothesis: str) -> dict:
    if not reference or not hypothesis:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(reference.lower(), hypothesis.lower())
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def extract_reference_diagnoses(prompt: str) -> str:
    """Extract the patient's actual diagnoses list from the prompt."""
    if not isinstance(prompt, str): return ""
    match = re.search(r"== DIAGNOSES \(Primary \+ Comorbidities\) ==\n(.*?)\n\n==", prompt, re.DOTALL)
    if match:
        raw_dx = match.group(1).strip()
        # Clean up '1. ', '2. ', and '[Primary]' tags to just leave clinical text
        raw_dx = re.sub(r"^\s*\d+\.\s*", "", raw_dx, flags=re.MULTILINE)
        raw_dx = raw_dx.replace("[Primary]", "").strip()
        return raw_dx
    return ""

def extract_hypothesis_diagnoses(output: str) -> str:
    """Extract only the '(1) Additional Diagnosis Hypotheses' section from the model output."""
    if not isinstance(output, str): return ""
    # We look for the text between (1) and (2). If (2) is missing, we take till the end.
    match = re.search(r"\(1\)\s*Additional Diagnosis Hypotheses\n(.*?)(?:\n\s*\n\(2\)|\Z)", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

# -------------- MAIN --------------------
def main():
    print("Loading batch results...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found!")
        return

    records = []
    total_rows = len(df)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"\rProcessing patient {i}/{total_rows}...", end="", flush=True)
        
        prompt = str(row.get("prompt", ""))
        ref = extract_reference_diagnoses(prompt)
        
        # If we somehow failed to parse the prompt, fallback to primary_dx
        if not ref:
            ref = str(row.get("primary_dx", ""))
            
        if not ref:
            continue

        entry = {
            "subject_id": row.get("subject_id", ""),
            "hadm_id":    row.get("hadm_id", ""),
            "all_diagnoses_ref": ref,
        }

        for model_label, col in MODEL_COLS.items():
            full_hyp = str(row.get(col, "")) if pd.notna(row.get(col)) else ""
            hyp_dx = extract_hypothesis_diagnoses(full_hyp)
            
            # If parsing failed, fallback to full text (to avoid failing silently)
            if not hyp_dx:
                hyp_dx = full_hyp

            bleu = compute_bleu(ref, hyp_dx)
            rouge = compute_rouge(ref, hyp_dx)
            
            entry[f"{model_label}_hyp_dx"] = hyp_dx
            entry[f"{model_label}_bleu"]   = round(bleu, 4)
            entry[f"{model_label}_rouge1"] = round(rouge["rouge1"], 4)
            entry[f"{model_label}_rouge2"] = round(rouge["rouge2"], 4)
            entry[f"{model_label}_rougeL"] = round(rouge["rougeL"], 4)

        records.append(entry)

    print() # Move to next line

    if not records:
        print("No valid records found to process.")
        return

    results_df = pd.DataFrame(records)
    
    # We drop the raw text columns from CSV so it doesn't get massive and messy.
    csv_df = results_df.copy()
    for model_label in MODEL_COLS:
        if f"{model_label}_hyp_dx" in csv_df.columns:
            csv_df = csv_df.drop(columns=[f"{model_label}_hyp_dx"])
    
    csv_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # -------------- SUMMARY --------------------
    print(f"\n{'MODEL':<20} {'BLEU':>8} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9}")
    print("-" * 58)
    for model_label in MODEL_COLS:
        b  = results_df[f"{model_label}_bleu"].mean()
        r1 = results_df[f"{model_label}_rouge1"].mean()
        r2 = results_df[f"{model_label}_rouge2"].mean()
        rl = results_df[f"{model_label}_rougeL"].mean()
        print(f"{model_label:<20} {b:>8.4f} {r1:>9.4f} {r2:>9.4f} {rl:>9.4f}")

    print(f"\n✅ Saved → {OUTPUT_CSV}  ({len(results_df)} patient)")

if __name__ == "__main__":
    main()
