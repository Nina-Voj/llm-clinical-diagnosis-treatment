# ============================================================
# BSP2 / Project Omega -- Step 4: Dashboard & Visualizations
# 
# RUN ORDER:
# 1. patients_pipeline.py     (Generates LLM responses & evaluator scores)
# 2. bleu_rouge_metrics.py      (Calculates text similarity metrics)
# 3. temperature_test.py        (Runs the hyperparameter experiment)
# 4. graph_maker.py             <-- THIS FILE (Builds the main visualization dashboard)
# 5. lime_analyzer.py           (Runs LIME explainability analysis)
# ============================================================
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ------- CSV READING ----------
df   = pd.read_csv("mimic_batch_results.csv")
br   = pd.read_csv("bleu_rouge_results.csv")
temp = pd.read_csv("temperature_results.csv")
ev   = pd.read_csv("mimic_batch_evaluations.csv")

# ---------- DRUG OVERLAP RATIOS ----------
df["total_true_drugs"] = df["true_rx_count"].clip(lower=1)
df["model1_rx_ratio"]  = df["model1_overlap"] / df["total_true_drugs"]
df["model2_rx_ratio"]  = df["model2_overlap"] / df["total_true_drugs"]

# ---------- TEMPERATURE → ROUGE-1 & BLEU ----------
r_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
smooth   = SmoothingFunction().method1
br_ref   = br[["hadm_id", "all_diagnoses_ref"]].drop_duplicates()
tm       = temp.merge(br_ref, on="hadm_id", how="left")

tm_rows = []
total_tm = len(tm)
for i, (_, row) in enumerate(tm.iterrows(), 1):
    print(f"\rPreparing graph data {i}/{total_tm}...", end="", flush=True)
    ref = str(row.get("all_diagnoses_ref", "")) if pd.notna(row.get("all_diagnoses_ref")) else ""
    hyp = str(row.get("output", ""))      if pd.notna(row.get("output"))     else ""
    if not ref or not hyp:
        continue
    r1   = r_scorer.score(ref.lower(), hyp.lower())["rouge1"].fmeasure
    rtok = nltk.word_tokenize(ref.lower())
    htok = nltk.word_tokenize(hyp.lower())
    bleu = sentence_bleu([rtok], htok, smoothing_function=smooth) if htok else 0.0
    tm_rows.append({"temperature": row["temperature"], "rouge1": r1, "bleu": bleu,
                    "output_length": row["output_length"]})
print()
tm_df = pd.DataFrame(tm_rows)

# ---------- COLORS & LABELS ----------
colors      = {"Llama-70B": "#4C72B0", "Gemma-31B": "#DD8452"}
temp_colors = {"0.1": "#E1B12C", "0.5": "#C0392B", "1.0": "#8D6E63"}

# ---------- 3 row × 3 column LAYOUT ----------
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        # Row 1 – eski paneller
        "Output Length (chars)", "Drug Overlap (count)", "Drug Overlap (%)",
        # Row 2 – BLEU/ROUGE + temperature
        "BLEU Score", "ROUGE-1 Score", "Temp: Output Length (chars)",
        # Row 3 – Evaluator skorları
        "Evaluator: Total Score (%)", "Evaluator: Accuracy (%)", "Evaluator: Safety (%)",
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.18,
)

# ---------- ROW 1: old 2 PANEL ----------
fig.add_trace(go.Box(y=df["model1_len"], name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=True, boxpoints=False),  row=1, col=1)
fig.add_trace(go.Box(y=df["model2_len"], name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=True, boxpoints=False),  row=1, col=1)

fig.add_trace(go.Box(y=df["model1_overlap"], name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=False, boxpoints=False), row=1, col=2)
fig.add_trace(go.Box(y=df["model2_overlap"], name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=False, boxpoints=False), row=1, col=2)

fig.add_trace(go.Box(y=df["model1_rx_ratio"], name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=False, boxpoints=False), row=1, col=3)
fig.add_trace(go.Box(y=df["model2_rx_ratio"], name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=False, boxpoints=False), row=1, col=3)

# ---------- ROW 2: BLEU / ROUGE / TEMPERATURE ----------
models      = ["llama-3.3-70b", "gemma-4-31b"]
model_names = ["Llama-70B",     "Gemma-31B"]

for label, mcol in zip(model_names, models):
    fig.add_trace(
        go.Bar(x=[label], y=[br[f"{mcol}_bleu"].mean()],
               name=label, marker_color=colors[label], showlegend=False,
               text=[f"{br[f'{mcol}_bleu'].mean():.4f}"], textposition="outside"),
        row=2, col=1)

for label, mcol in zip(model_names, models):
    fig.add_trace(
        go.Bar(x=[label], y=[br[f"{mcol}_rouge1"].mean()],
               name=label, marker_color=colors[label], showlegend=False,
               text=[f"{br[f'{mcol}_rouge1'].mean():.4f}"], textposition="outside"),
        row=2, col=2)

for t in [0.1, 0.5, 1.0]:
    sub = tm_df[tm_df["temperature"] == t]
    fig.add_trace(
        go.Box(y=sub["output_length"], name=f"temp={t}",
               marker_color=temp_colors[str(t)], showlegend=True, boxpoints=False),
        row=2, col=3)

# ---------- ROW 3: EVALUATOR Scores ----------
# Sanity check assertions to prevent silent scaling errors if evaluator format changes
for col in ["model1_total", "model2_total"]:
    if col in ev.columns and not ev[col].empty:
        assert ev[col].max() <= 100.0, f"Error: {col} exceeds 100. Check evaluator scoring scale!"
        
for col in ["model1_accuracy", "model2_accuracy", "model1_safety", "model2_safety"]:
    if col in ev.columns and not ev[col].empty:
        assert ev[col].max() <= 25.0, f"Error: {col} exceeds 25. Check evaluator scoring scale!"

# Total Score (divide by 100 to get percentage)
fig.add_trace(go.Box(y=ev["model1_total"] / 100.0, name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=False, boxpoints=False), row=3, col=1)
fig.add_trace(go.Box(y=ev["model2_total"] / 100.0, name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=False, boxpoints=False), row=3, col=1)

# Accuracy (divide by 25 to get percentage)
fig.add_trace(go.Box(y=ev["model1_accuracy"] / 25.0, name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=False, boxpoints=False), row=3, col=2)
fig.add_trace(go.Box(y=ev["model2_accuracy"] / 25.0, name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=False, boxpoints=False), row=3, col=2)

# Safety (divide by 25 to get percentage)
fig.add_trace(go.Box(y=ev["model1_safety"] / 25.0, name="Llama-70B",    marker_color=colors["Llama-70B"],    showlegend=False, boxpoints=False), row=3, col=3)
fig.add_trace(go.Box(y=ev["model2_safety"] / 25.0, name="Gemma-31B",    marker_color=colors["Gemma-31B"],    showlegend=False, boxpoints=False), row=3, col=3)

# ---------- LAYOUT & AXES RANGES ----------
# Calculate dynamic maximums with padding so nothing is cut off
max_len = max(df["model1_len"].max(), df["model2_len"].max())
max_overlap = max(df["model1_overlap"].max(), df["model2_overlap"].max())
max_ratio = max(df["model1_rx_ratio"].max(), df["model2_rx_ratio"].max())

bleu_means = [br[f"{mcol}_bleu"].mean() for mcol in models]
max_bleu_mean = max(bleu_means) if bleu_means else 0.1
rouge_means = [br[f"{mcol}_rouge1"].mean() for mcol in models]
max_rouge_mean = max(rouge_means) if rouge_means else 0.1

max_temp_len = tm_df["output_length"].max() if not tm_df.empty else 1000

max_eval_total = max(ev["model1_total"].max(), ev["model2_total"].max())
max_eval_acc = max(ev["model1_accuracy"].max(), ev["model2_accuracy"].max())
max_eval_safety = max(ev["model1_safety"].max(), ev["model2_safety"].max())

fig.update_layout(
    title=dict(
        text="Project Omega – Full Evaluation Dashboard (Llama-70B vs Gemma-31B)",
        x=0.5, xanchor="center", font=dict(size=22),
    ),
    width=1200,
    height=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=14)),
    margin=dict(t=120, b=50, l=50, r=30),
    template="plotly_white",
    barmode="group",
)

# Apply ranges (start at 0, pad max by 15-30%)
fig.update_yaxes(range=[0, max_len * 1.15], row=1, col=1)
fig.update_yaxes(range=[0, max_overlap + 1 if max_overlap < 5 else max_overlap * 1.15], row=1, col=2)
fig.update_yaxes(range=[0, max(1.0, max_ratio) * 1.15], tickformat=".0%", row=1, col=3)

fig.update_yaxes(range=[0, max_bleu_mean * 1.3], row=2, col=1)
fig.update_yaxes(range=[0, max_rouge_mean * 1.3], row=2, col=2)
fig.update_yaxes(range=[0, max_temp_len * 1.15], row=2, col=3)

fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=3, col=1)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=3, col=2)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=3, col=3)

try:
    fig.write_image("model_comparison_chart.png", scale=2)
    print("saved -> model_comparison_chart.png (High Resolution)")
except ValueError as e:
    if "kaleido" in str(e).lower():
        print("ERROR: kaleido package is required for static image export.")
        print("Please run: pip install -U kaleido")
    else:
        print(f"couldn't save PNG ({e})")
except Exception as e:
    print(f"couldn't save PNG ({e})")
