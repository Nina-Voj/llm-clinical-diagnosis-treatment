# ============================================================
# BSP2 / Project Omega -- Step 4: Dashboard & Visualizations
# 
# RUN ORDER:
# 1. patients_pipeline.py       (Generates LLM responses & evaluator scores)
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

# ---------- CONFIG & CONSTANTS ----------
MAX_TOTAL    = 100.0   # evaluator total score ceiling
MAX_SUBSCORE =  25.0   # ceiling for each evaluator subscore (accuracy/safety/completeness/usefulness)

# ------- CSV READING ----------
df   = pd.read_csv("mimic_batch_results.csv")
br   = pd.read_csv("bleu_rouge_results.csv")
temp = pd.read_csv("temperature_results.csv")
ev   = pd.read_csv("mimic_batch_evaluations.csv")

# ---------- DRUG OVERLAP RATIOS ----------
df["total_true_drugs"] = df["true_rx_count"].clip(lower=1)
df["model1_rx_ratio"]  = df["model1_overlap"] / df["total_true_drugs"]
df["model2_rx_ratio"]  = df["model2_overlap"] / df["total_true_drugs"]

# ---------- TEMPERATURE --> ROUGE-1 & BLEU ----------
# Compute per-row BLEU & ROUGE-1 for every temperature experiment entry.
# Reference: the actual patient diagnoses from bleu_rouge_results.csv.
# Note: lexical-overlap proxy, not a live model evaluation.
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

# N per temperature (for subtitle labelling)
n_per_temp = tm_df.groupby("temperature").size().to_dict()

# ---------- WINNER BAR CHART DATA ----------
winner_counts = ev["winner"].value_counts()
winner_labels = ["Llama-70B Wins", "Gemma-31B Wins", "Tie"]
winner_keys   = ["model1",         "model2",          "tie"]
winner_values = [int(winner_counts.get(k, 0)) for k in winner_keys]
winner_colors = ["#4C72B0", "#DD8452", "#9E9E9E"]

# ---------- COLORS & LABELS ----------
colors      = {"Llama-70B": "#4C72B0", "Gemma-31B": "#DD8452"}
temp_colors = {"0.1": "#E1B12C", "0.5": "#C0392B", "1.0": "#8D6E63"}
models      = ["llama-3.3-70b", "gemma-4-31b"]
model_names = ["Llama-70B",     "Gemma-31B"]

# ---------- SANITY CHECKS ----------
for col in ["model1_total", "model2_total"]:
    if col in ev.columns and not ev[col].empty:
        assert ev[col].max() <= MAX_TOTAL, f"Error: {col} exceeds {MAX_TOTAL}. Check evaluator scoring scale!"

for col in ["model1_accuracy", "model2_accuracy", "model1_safety", "model2_safety",
            "model1_completeness", "model2_completeness", "model1_usefulness", "model2_usefulness"]:
    if col in ev.columns and not ev[col].empty:
        assert ev[col].max() <= MAX_SUBSCORE, f"Error: {col} exceeds {MAX_SUBSCORE}. Check evaluator scoring scale!"

# ---------- 5 row x 3 column LAYOUT ----------
# Row 1: Output stats           Length, Drug Overlap count, Drug Overlap %
# Row 2: Text similarity        BLEU, ROUGE-1, ROUGE-2
# Row 3: Text similarity cont.  ROUGE-L, Evaluator Total, Winner bar chart
# Row 4: Evaluator subscores    Accuracy, Safety, Completeness
# Row 5: Evaluator + Temp       Usefulness, Temp-->BLEU, Temp-->ROUGE-1
fig = make_subplots(
    rows=5, cols=3,
    subplot_titles=[
        # Row 1
        f"Output Length (chars) (n={len(df)})",
        f"Drug Overlap — count (n={len(df)})",
        f"Drug Overlap — % of True Rx (n={len(df)})",
        # Row 2
        f"BLEU Score (n={len(br)})",
        f"ROUGE-1 F1 (n={len(br)})",
        f"ROUGE-2 F1 (n={len(br)})",
        # Row 3
        f"ROUGE-L F1 (n={len(br)})",
        f"Evaluator: Total Score % (n={len(ev)})",
        f"Head-to-Head Winner (n={len(ev)})",
        # Row 4
        f"Evaluator: Accuracy % (n={len(ev)})",
        f"Evaluator: Safety % (n={len(ev)})",
        f"Evaluator: Completeness % (n={len(ev)})",
        # Row 5
        f"Evaluator: Usefulness % (n={len(ev)})",
        f"Temp → BLEU (Gemma-31B, n={n_per_temp.get(0.1, '?')}/temp)",
        f"Temp → ROUGE-1 (Gemma-31B, n={n_per_temp.get(0.1, '?')}/temp)",
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.10,
)

# ── ROW 1: Output Stats ──────────────────────────────────────────────
fig.add_trace(go.Box(y=df["model1_len"],      name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=True,  boxpoints=False), row=1, col=1)
fig.add_trace(go.Box(y=df["model2_len"],      name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=True,  boxpoints=False), row=1, col=1)
fig.add_trace(go.Box(y=df["model1_overlap"],  name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=1, col=2)
fig.add_trace(go.Box(y=df["model2_overlap"],  name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=1, col=2)
fig.add_trace(go.Box(y=df["model1_rx_ratio"], name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=1, col=3)
fig.add_trace(go.Box(y=df["model2_rx_ratio"], name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=1, col=3)

# ── ROW 2: BLEU / ROUGE-1 / ROUGE-2 ─────────────────────────────────
for label, mcol in zip(model_names, models):
    fig.add_trace(go.Box(y=br[f"{mcol}_bleu"],   name=label, marker_color=colors[label], showlegend=False, boxpoints=False), row=2, col=1)
for label, mcol in zip(model_names, models):
    fig.add_trace(go.Box(y=br[f"{mcol}_rouge1"], name=label, marker_color=colors[label], showlegend=False, boxpoints=False), row=2, col=2)
for label, mcol in zip(model_names, models):
    fig.add_trace(go.Box(y=br[f"{mcol}_rouge2"], name=label, marker_color=colors[label], showlegend=False, boxpoints=False), row=2, col=3)

# ── ROW 3: ROUGE-L / Evaluator Total / Winner Chart ──────────────────
for label, mcol in zip(model_names, models):
    fig.add_trace(go.Box(y=br[f"{mcol}_rougeL"], name=label, marker_color=colors[label], showlegend=False, boxpoints=False), row=3, col=1)

fig.add_trace(go.Box(y=ev["model1_total"] / MAX_TOTAL, name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=3, col=2)
fig.add_trace(go.Box(y=ev["model2_total"] / MAX_TOTAL, name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=3, col=2)

fig.add_trace(go.Bar(
    x=winner_labels, y=winner_values,
    marker_color=winner_colors,
    text=winner_values, textposition="outside",
    showlegend=False,
), row=3, col=3)

# -- ROW 4: Evaluator Accuracy / Safety / Completeness ----------------
fig.add_trace(go.Box(y=ev["model1_accuracy"]     / MAX_SUBSCORE, name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=4, col=1)
fig.add_trace(go.Box(y=ev["model2_accuracy"]     / MAX_SUBSCORE, name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=4, col=1)
fig.add_trace(go.Box(y=ev["model1_safety"]       / MAX_SUBSCORE, name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=4, col=2)
fig.add_trace(go.Box(y=ev["model2_safety"]       / MAX_SUBSCORE, name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=4, col=2)
fig.add_trace(go.Box(y=ev["model1_completeness"] / MAX_SUBSCORE, name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=4, col=3)
fig.add_trace(go.Box(y=ev["model2_completeness"] / MAX_SUBSCORE, name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=4, col=3)

# -- ROW 5: Usefulness / Temperature BLEU / Temperature ROUGE-1 ------------
fig.add_trace(go.Box(y=ev["model1_usefulness"] / MAX_SUBSCORE, name="Llama-70B", marker_color=colors["Llama-70B"], showlegend=False, boxpoints=False), row=5, col=1)
fig.add_trace(go.Box(y=ev["model2_usefulness"] / MAX_SUBSCORE, name="Gemma-31B", marker_color=colors["Gemma-31B"], showlegend=False, boxpoints=False), row=5, col=1)

# Temperature quality panels, each box = n=1000 patients at one temperature.
# Answers: "Does higher temperature degrade lexical quality?"
for t in [0.1, 0.5, 1.0]:
    sub = tm_df[tm_df["temperature"] == t]
    tc  = temp_colors[str(t)]
    fig.add_trace(go.Box(y=sub["bleu"],   name=f"temp={t}", marker_color=tc, showlegend=True,  boxpoints=False, legendgroup=f"t{t}"), row=5, col=2)
    fig.add_trace(go.Box(y=sub["rouge1"], name=f"temp={t}", marker_color=tc, showlegend=False, boxpoints=False, legendgroup=f"t{t}"), row=5, col=3)

# ---------- LAYOUT & AXES RANGES ----------
max_len     = max(df["model1_len"].max(),     df["model2_len"].max())
max_overlap = max(df["model1_overlap"].max(), df["model2_overlap"].max())
max_ratio   = max(df["model1_rx_ratio"].max(),df["model2_rx_ratio"].max())

bleu_maxs    = [br[f"{mcol}_bleu"].max()   for mcol in models if f"{mcol}_bleu"   in br.columns and not br[f"{mcol}_bleu"].dropna().empty]
max_bleu_val = max(bleu_maxs)  if bleu_maxs  else 0.1
rouge1_maxs  = [br[f"{mcol}_rouge1"].max() for mcol in models if f"{mcol}_rouge1" in br.columns and not br[f"{mcol}_rouge1"].dropna().empty]
max_rouge1   = max(rouge1_maxs) if rouge1_maxs else 0.1
rouge2_maxs  = [br[f"{mcol}_rouge2"].max() for mcol in models if f"{mcol}_rouge2" in br.columns and not br[f"{mcol}_rouge2"].dropna().empty]
max_rouge2   = max(rouge2_maxs) if rouge2_maxs else 0.1
rougeL_maxs  = [br[f"{mcol}_rougeL"].max() for mcol in models if f"{mcol}_rougeL" in br.columns and not br[f"{mcol}_rougeL"].dropna().empty]
max_rougeL   = max(rougeL_maxs) if rougeL_maxs else 0.1

max_temp_bleu  = tm_df["bleu"].max()   if not tm_df.empty else 0.1
max_temp_rouge = tm_df["rouge1"].max() if not tm_df.empty else 0.1

fig.update_layout(
    title=dict(
        text="Project Omega – Full Evaluation Dashboard (Llama-70B vs Gemma-31B)",
        x=0.5, xanchor="center", font=dict(size=22),
    ),
    width=1200,
    height=1750,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
    margin=dict(t=120, b=60, l=50, r=30),
    template="plotly_white",
    barmode="group",
)

# Row 1: output stats
fig.update_yaxes(range=[0, max_len * 1.15], row=1, col=1)
fig.update_yaxes(range=[0, max_overlap + 1 if max_overlap < 5 else max_overlap * 1.15], row=1, col=2)
fig.update_yaxes(range=[0, max(1.0, max_ratio) * 1.15], tickformat=".0%", row=1, col=3)

# Row 2: BLEU / ROUGE-1 / ROUGE-2
fig.update_yaxes(range=[0, max_bleu_val * 1.3], row=2, col=1)
fig.update_yaxes(range=[0, max_rouge1   * 1.3], row=2, col=2)
fig.update_yaxes(range=[0, max_rouge2   * 1.3], row=2, col=3)

# Row 3: ROUGE-L / Total % / Winner (winner y auto)
fig.update_yaxes(range=[0, max_rougeL * 1.3], row=3, col=1)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=3, col=2)
fig.update_yaxes(range=[0, max(winner_values) * 1.25], row=3, col=3)

# Row 4: Accuracy / Safety / Completeness (all %)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=4, col=1)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=4, col=2)
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=4, col=3)

# Row 5: Usefulness / Temp-BLEU / Temp-ROUGE-1
fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=5, col=1)
fig.update_yaxes(range=[0, max_temp_bleu  * 1.3], row=5, col=2)
fig.update_yaxes(range=[0, max_temp_rouge * 1.3], row=5, col=3)

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
