import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ==============================
# CONFIGURATION PAGE
# ==============================
st.set_page_config(
    page_title="Dashboard PFE SMA",
    layout="wide",
    page_icon="🎓"
)

# ==============================
# STYLE CSS
# ==============================
st.markdown("""
<style>
.big-title {
    font-size: 28px;
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 4px;
}
.sub-info {
    font-size: 13px;
    color: #888;
    margin-bottom: 20px;
}
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 14px;
}
.hyp-box {
    background-color: #f8f9fa;
    border-left: 4px solid #1f77b4;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title"> Dashboard — Simulation Multi-Agents</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-info">PFE 2024–2025 | Modélisation de l\'apprentissage collaboratif avec SMA + IA</p>', unsafe_allow_html=True)

# ==============================
# CHARGEMENT DONNÉES
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/results.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Fichier data/results.csv introuvable. Lance d'abord run.py pour générer les données.")
    st.info("Commande : python run.py")
    st.stop()

# ==============================
# MAPPING DES SCÉNARIOS
# ==============================
COULEURS = {
    "S1 — Sans tuteur"         : "#2196F3",   # bleu
    "S2 — Avec tuteur IA"      : "#FF9800",   # orange
    "S3 — Groupes intelligents": "#4CAF50",   # vert
    "S4 — Auto-apprentissage"  : "#F44336",   # rouge
}

def get_scenario_name(row):
    """Détermine le nom du scénario selon les colonnes disponibles."""
    auto = row.get("auto_learning", False)
    if auto:
        return "S4 — Auto-apprentissage"
    smart = row.get("smart_group", False)
    tutor = row.get("tutor", False)
    if smart:
        # S3 = groupes intelligents (sans tuteur dans le design final)
        return "S3 — Groupes intelligents"
    elif tutor:
        return "S2 — Avec tuteur IA"
    else:
        return "S1 — Sans tuteur"

df["scenario"] = df.apply(get_scenario_name, axis=1)

# ==============================
# SIDEBAR — Filtres
# ==============================
st.sidebar.header(" Filtres")

scenario_options = sorted(df["scenario"].unique().tolist())

selected_scenario = st.sidebar.multiselect(
    "Choisir les scénarios",
    options=scenario_options,
    default=scenario_options
)

run_min = int(df["run"].min())
run_max = int(df["run"].max())

selected_runs = st.sidebar.slider(
    "Intervalle des runs",
    min_value=run_min,
    max_value=run_max,
    value=(run_min, run_max)
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Légende des scénarios**")
st.sidebar.markdown(" S1 — Sans tuteur (groupe contrôle)")
st.sidebar.markdown(" S2 — Avec tuteur IA (Random Forest)")
st.sidebar.markdown(" S3 — Groupes intelligents (K-Means)")
st.sidebar.markdown(" S4 — Auto-apprentissage adaptatif")

# ==============================
# FILTRAGE
# ==============================
df_filtered = df[
    (df["scenario"].isin(selected_scenario)) &
    (df["run"] >= selected_runs[0]) &
    (df["run"] <= selected_runs[1])
].copy()

if df_filtered.empty:
    st.warning("⚠️ Aucune donnée avec ces filtres.")
    st.stop()

# ==============================
# CALCUL DES KPI (UN SEUL ENDROIT — plus de duplication)
# ==============================
perf_mean = df_filtered["performance"].mean()
std_perf = df_filtered["performance"].std()

# p_drop : c'est un label binaire (0 ou 1), donc la moyenne = proportion à risque
if "p_drop" in df_filtered.columns:
    p_drop_mean = float(df_filtered["p_drop"].mean())
else:
    p_drop_mean = None

# Interventions : MOYENNE par run
# Comme la valeur est dupliquée sur les lignes du même run (1 valeur par run
# × N étudiants), on prend la première valeur de chaque run puis la moyenne
if "interventions" in df_filtered.columns:
    interventions_mean = float(
        df_filtered.groupby(["scenario", "run"])["interventions"].first().mean()
    )
else:
    interventions_mean = 0.0

# ==============================
# AFFICHAGE DES KPI
# ==============================
st.subheader(" Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Performance moyenne",
    value=round(perf_mean, 3),
    delta=("↑ Bon" if perf_mean > 0.6 else "↓ Faible")
)
col2.metric(
    label="Écart-type",
    value=round(std_perf, 3),
    delta=("⚠ Inégalités élevées" if std_perf > 0.3 else "✓ Stable")
)

if p_drop_mean is not None:
    if p_drop_mean < 0.3:
        label_pdrop = " Faible"
    elif p_drop_mean < 0.6:
        label_pdrop = " Moyen"
    else:
        label_pdrop = " Élevé"
    col3.metric(
        label="Taux d'étudiants à risque",
        value=f"{round(p_drop_mean * 100, 1)}%",
        delta=label_pdrop
    )
else:
    col3.metric(label="Taux d'étudiants à risque", value="N/A")

col4.metric(
    label="Interventions tuteur (moy./run)",
    value=round(interventions_mean, 1)
)

st.markdown("---")

# ==============================
# GRAPHE 1 : Performance moyenne par scénario AVEC IC 95%
# ==============================
st.subheader(" Performance moyenne par scénario (avec IC à 95%)")

scenario_stats = []
for sc in df_filtered["scenario"].unique():
    perfs = df_filtered[df_filtered["scenario"] == sc]["performance"].values
    n = len(perfs)
    mean = np.mean(perfs)
    if n >= 2:
        sem = stats.sem(perfs)
        ci_hw = sem * stats.t.ppf(0.975, n - 1)
    else:
        ci_hw = 0
    scenario_stats.append({"scenario": sc, "mean": mean, "ci": ci_hw, "n": n})

stats_df = pd.DataFrame(scenario_stats).sort_values("mean")

fig1, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.bar(
    stats_df["scenario"],
    stats_df["mean"],
    yerr=stats_df["ci"],
    capsize=8,
    color=[COULEURS.get(sc, "#999999") for sc in stats_df["scenario"]],
    edgecolor="white",
    linewidth=0.8,
    width=0.55,
    error_kw={"linewidth": 1.5, "ecolor": "black"}
)

ax1.axhline(y=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Seuil réussite (0.75)")

for bar, mean, ci in zip(bars, stats_df["mean"], stats_df["ci"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        mean + ci + 0.02,
        f"{mean:.3f}\n±{ci:.3f}",
        ha="center", va="bottom",
        fontsize=9, fontweight="bold"
    )

ax1.set_ylabel("Performance moyenne", fontsize=11)

# CORRECTION : on compte le nombre de RUNS UNIQUES par scénario,
# pas le nombre de lignes du CSV (qui est runs × étudiants).
n_runs_per_scenario = int(
    df_filtered.groupby("scenario")["run"].nunique().mean()
)
n_students_per_run = int(
    df_filtered.groupby(["scenario", "run"]).size().mean()
)
ax1.set_title(
    f"Comparaison des scénarios "
    f"(n = {n_runs_per_scenario} runs × {n_students_per_run} étudiants par scénario)",
    fontsize=13, fontweight="bold"
)
ax1.set_ylim(0, 1.15)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()

st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ==============================
# GRAPHE 2 : Évolution par run
# ==============================
st.subheader(" Évolution de la performance par run")

fig2, ax2 = plt.subplots(figsize=(10, 5))
for scenario in df_filtered["scenario"].unique():
    subset = df_filtered[df_filtered["scenario"] == scenario]
    evolution = subset.groupby("run")["performance"].mean()
    couleur = COULEURS.get(scenario, "#999999")
    ax2.plot(
        evolution.index, evolution.values,
        marker="o", markersize=4, linewidth=2.2,
        color=couleur, label=scenario
    )

ax2.axhline(y=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Seuil réussite")
ax2.set_xlabel("Run", fontsize=11)
ax2.set_ylabel("Performance moyenne", fontsize=11)
ax2.set_title("Évolution de la performance par run et par scénario", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ==============================
# GRAPHE 3 : Distribution
# ==============================
st.subheader(" Distribution des performances")

fig3, ax3 = plt.subplots(figsize=(9, 4))
for scenario in df_filtered["scenario"].unique():
    subset = df_filtered[df_filtered["scenario"] == scenario]
    couleur = COULEURS.get(scenario, "#999999")
    ax3.hist(
        subset["performance"], bins=15, alpha=0.55,
        color=couleur, label=scenario, edgecolor="white"
    )

ax3.axvline(x=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Seuil réussite")
ax3.set_title("Distribution des performances par scénario", fontsize=13, fontweight="bold")
ax3.set_xlabel("Performance", fontsize=11)
ax3.set_ylabel("Fréquence", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.markdown("---")

# ==============================
# TABLEAU STATISTIQUES DÉTAILLÉES
# ==============================
st.subheader(" Statistiques détaillées par scénario")

cols_agg = {"performance": ["mean", "std", "min", "max"]}
if "p_drop" in df_filtered.columns:
    cols_agg["p_drop"] = ["mean"]

summary = df_filtered.groupby("scenario").agg(cols_agg).round(3)
summary.columns = (
    ["Perf. moyenne", "Écart-type", "Perf. min", "Perf. max"]
    + (["Taux à risque"] if "p_drop" in df_filtered.columns else [])
)

# Interventions : 1 valeur par run, donc on prend .first() puis on moyenne
if "interventions" in df_filtered.columns:
    interv_per_scenario = (
        df_filtered.groupby(["scenario", "run"])["interventions"]
        .first()
        .groupby("scenario")
        .mean()
        .round(2)
    )
    summary["Interventions (moy./run)"] = interv_per_scenario

st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ==============================
# VALIDATION DES HYPOTHÈSES (CORRIGÉE - les 5 hypothèses avec tests stats)
# ==============================
st.subheader(" Validation des hypothèses de recherche")
st.markdown(
    "_Tests statistiques basés sur le t-test de Welch (α = 0.05) "
    "et la corrélation de Pearson._"
)

perf_par_sc = df_filtered.groupby("scenario")["performance"].mean()


def get_perfs(scenario_name):
    """Récupère les perfs d'un scénario, ou None s'il est absent."""
    if scenario_name in df_filtered["scenario"].values:
        return df_filtered[df_filtered["scenario"] == scenario_name]["performance"].values
    return None


def status_label(valid, partial=False):
    if partial:
        return " PARTIELLEMENT VALIDÉE"
    return "✅ VALIDÉE" if valid else "❌ NON VALIDÉE"


# ----- Hypotheses corrigees (lues depuis data/hypotheses.json) -----
# Calculees par run_analysis.py (qui exporte data/hypotheses.json).
# Le dashboard se contente d'AFFICHER ces resultats (aucune simulation ici).
import json as _json
import os as _os

_hyp_path = "data/hypotheses.json"
if _os.path.exists(_hyp_path):
    with open(_hyp_path, encoding="utf-8") as _f:
        HYP = _json.load(_f)

    def _statut(v):
        return "\u2705 VALID\u00c9E" if v else "\u274c NON VALID\u00c9E"

    # H1 - collaboration
    h = HYP["H1"]
    st.markdown("#### H1 \u2014 La collaboration am\u00e9liore la performance collective")
    c1, c2, c3 = st.columns(3)
    c1.metric(h["metrique"], h["valeur"], f"{h['gain_pct']:+.1f}%")
    c2.metric("p-value", f"{h['p_value']:.2e}")
    c3.metric("Statut", _statut(h["valide"]))
    st.caption(
        "Test : performance collective d'un sc\u00e9nario COLLABORATIF (avec entraide "
        "entre pairs) vs un sc\u00e9nario INDIVIDUEL (sans entraide), sur 50 runs. "
        "Cette comparaison isole l'effet causal de la collaboration, contrairement "
        f"\u00e0 une simple corr\u00e9lation biais\u00e9e par la demande d'aide. Cohen's d = {h['cohens_d']} (grand effet)."
    )
    st.markdown("")

    # H2 - tuteur IA
    h = HYP["H2"]
    st.markdown("#### H2 \u2014 L'intervention du tuteur IA am\u00e9liore la performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(h["metrique"], h["valeur"], f"{h['gain_pct']:+.1f}%")
    c2.metric("p-value", f"{h['p_value']:.2e}")
    c3.metric("Cohen's d", f"{h['cohens_d']}")
    st.caption(f"Statut : {_statut(h['valide'])}. Comparaison S2 (avec tuteur) vs S1 (sans tuteur).")
    st.markdown("")

    # H3 - groupes intelligents
    h = HYP["H3"]
    st.markdown("#### H3 \u2014 Les groupes intelligents am\u00e9liorent la performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(h["metrique"], h["valeur"])
    c2.metric("p-value", f"{h['p_value']:.2e}")
    c3.metric("Statut", _statut(h["valide"]))
    st.caption(
        "Test : performance S3 (groupes homog\u00e8nes par niveau, entraide dans le groupe, "
        "sans tuteur) vs S1 (entraide dans toute la classe, sans tuteur). Les deux sans "
        f"tuteur pour isoler l'effet pur du groupement. Cohen's d = {h['cohens_d']}. "
        "R\u00e9sultat : le groupement intelligent n'am\u00e9liore pas la performance. En groupes "
        "homog\u00e8nes, les \u00e9tudiants faibles n'ont pas de co\u00e9quipier plus fort pour les aider, "
        "alors que l'entraide \u00e0 l'\u00e9chelle de la classe (S1) les atteint."
    )
    st.markdown("")

    # H4 (ex-H5) - Random Forest
    h = HYP["H4"]
    st.markdown("#### H4 \u2014 Utiliser l'IA pour d\u00e9tecter les \u00e9tudiants en difficult\u00e9")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{h['accuracy']:.3f}")
    c2.metric("F1-score", f"{h['f1']:.3f}")
    c3.metric("Recall", f"{h['recall']:.3f}")
    c4.metric("Statut", _statut(h["valide"]))
    st.caption(
        "Random Forest entra\u00een\u00e9 en excluant comp\u00e9tence et motivation des features (\u00e9vite "
        "le data leakage). Features : tutor, smart_group, interactions, charge, engagement."
    )
else:
    st.warning("data/hypotheses.json introuvable. Lance d'abord : python run_analysis.py")

st.markdown("---")

# ==============================
# TEST GLOBAL : ANOVA
# ==============================
st.subheader(" Test global de différence entre scénarios (ANOVA)")
groups_for_anova = []
labels = []
for sc in selected_scenario:
    g = get_perfs(sc)
    if g is not None and len(g) >= 2:
        groups_for_anova.append(g)
        labels.append(sc)

if len(groups_for_anova) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups_for_anova)
    sig = " SIGNIFICATIF" if p_anova < 0.05 else "❌ Non significatif"
    c1, c2, c3 = st.columns(3)
    c1.metric("F-statistic", f"{f_stat:.2f}")
    c2.metric("p-value", f"{p_anova:.2e}")
    c3.metric("Conclusion", sig)
    st.caption(
        f"L'ANOVA teste si **au moins un** scénario diffère significativement "
        f"des autres. Avec p = {p_anova:.2e}, les configurations testées "
        f"produisent bien des résultats statistiquement différents."
    )

st.markdown("---")

# ==============================
# INTERPRÉTATION AUTOMATIQUE
# ==============================
st.subheader(" Interprétation automatique")

best_scenario = perf_par_sc.idxmax() if not perf_par_sc.empty else "N/A"
worst_scenario = perf_par_sc.idxmin() if not perf_par_sc.empty else "N/A"

interpretation = f"""
- La **performance moyenne globale** sur les scénarios sélectionnés est de **{round(perf_mean, 3)}**.
- L'**écart-type de {round(std_perf, 3)}** reflète la variabilité entre scénarios — une valeur élevée indique que les configurations ont des impacts très différents.
- Le **scénario le plus performant** est **{best_scenario}** et le **moins performant** est **{worst_scenario}**.
- Le **tuteur IA** (scénarios S2 et S4) est le levier déterminant : il améliore nettement la performance.
- En revanche, le **groupement intelligent** (S3, sans tuteur) ne fait **pas** mieux que la collaboration libre dans toute la classe (S1) : il tend même à faire **légèrement moins bien**. En groupes homogènes par niveau, les étudiants faibles se retrouvent ensemble et n'ont plus de pair plus fort pour les aider, alors que l'entraide à l'échelle de la classe (S1) leur donne accès à des camarades plus avancés.
"""

if p_drop_mean is not None:
    interpretation += (
        f"\n- Le **taux d'étudiants à risque** (performance < 0.6) est de "
        f"**{round(p_drop_mean * 100, 1)}%** sur les scénarios sélectionnés. "
        f"Ce label sert de vérité terrain pour entraîner le modèle Random Forest."
    )

st.markdown(interpretation)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<center> PFE 2025–2026 — Simulation d'apprentissage collaboratif avec SMA + IA</center>",
    unsafe_allow_html=True
)