import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIGURATION PAGE
# ==============================
st.set_page_config(
    page_title="Dashboard PFE SMA",
    layout="wide"
)

# ==============================
# STYLE
# ==============================
st.markdown("""
<style>
.big-title {
    font-size: 30px;
    font-weight: bold;
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="big-title"> Dashboard - Simulation Multi-Agents</p>',
    unsafe_allow_html=True
)

# ==============================
# CHARGEMENT DONNÉES
# ==============================
try:
    df = pd.read_csv("data/results.csv")
except:
    st.error("Fichier data/results.csv introuvable.")
    st.stop()

# ==============================
# AJOUT SCÉNARIO
# ==============================
def scenario_name(row):
    auto_learning = row.get("auto_learning", False)

    if auto_learning:
        return "Auto-apprentissage adaptatif"

    elif (row["tutor"] == False) and (row["smart_group"] == False):
        return "Sans tuteur"

    elif (row["tutor"] == True) and (row["smart_group"] == False):
        return "Avec tuteur IA"

    elif (row["tutor"] == True) and (row["smart_group"] == True):
        return "Groupes intelligents"

    return "Autre"

df["scenario"] = df.apply(scenario_name, axis=1)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("🔎 Filtres")

scenario_options = df["scenario"].unique()

selected_scenario = st.sidebar.multiselect(
    "Choisir les scénarios",
    options=scenario_options,
    default=scenario_options
)

run_min = int(df["run"].min())
run_max = int(df["run"].max())

selected_runs = st.sidebar.slider(
    "Intervalle des runs",
    run_min,
    run_max,
    (run_min, run_max)
)

# ==============================
# FILTRAGE
# ==============================
df_filtered = df[
    (df["scenario"].isin(selected_scenario)) &
    (df["run"] >= selected_runs[0]) &
    (df["run"] <= selected_runs[1])
]

if df_filtered.empty:
    st.warning("Aucune donnée disponible avec ces filtres.")
    st.stop()

# ==============================
# KPI
# ==============================
st.subheader(" Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

perf_mean = df_filtered["performance"].mean()
std_perf = df_filtered["performance"].std()

# si colonne disponible
if "p_drop" in df_filtered.columns:
    p_drop = df_filtered["p_drop"].mean()
else:
    p_drop = None

# si colonne disponible
if "interventions" in df_filtered.columns:
    interventions = int(df_filtered["interventions"].sum())
else:
    interventions = 0

col1.metric("Performance moyenne", round(perf_mean, 3))
col2.metric("Écart-type", round(std_perf, 3))

if p_drop is not None:
    col3.metric("p_drop moyen", round(p_drop, 3))
else:
    col3.metric("p_drop moyen", "N/A")

col4.metric("Interventions tuteur", interventions)

st.markdown("---")

# ==============================
# GRAPHE BARRE
# ==============================
st.subheader(" Performance moyenne par scénario")

grouped = df_filtered.groupby("scenario")["performance"].mean().sort_values()

fig1, ax1 = plt.subplots(figsize=(8, 5))
grouped.plot(kind="bar", ax=ax1)

ax1.set_ylabel("Performance")
ax1.set_title("Comparaison des scénarios")

st.pyplot(fig1)

# ==============================
# GRAPHE EVOLUTION
# ==============================
st.subheader(" Évolution de la performance")

fig2, ax2 = plt.subplots(figsize=(10, 5))

for scenario in df_filtered["scenario"].unique():
    subset = df_filtered[df_filtered["scenario"] == scenario]

    evolution = subset.groupby("run")["performance"].mean()

    ax2.plot(
        evolution.index,
        evolution.values,
        marker="o",
        label=scenario
    )

ax2.set_xlabel("Run")
ax2.set_ylabel("Performance")
ax2.set_title("Évolution moyenne par scénario")
ax2.legend()

st.pyplot(fig2)

# ==============================
# HISTOGRAMME
# ==============================
st.subheader(" Distribution des performances")

fig3, ax3 = plt.subplots(figsize=(8, 5))

for scenario in df_filtered["scenario"].unique():
    subset = df_filtered[df_filtered["scenario"] == scenario]
    ax3.hist(
        subset["performance"],
        bins=15,
        alpha=0.5,
        label=scenario
    )

ax3.set_title("Histogramme des performances")
ax3.set_xlabel("Performance")
ax3.set_ylabel("Fréquence")
ax3.legend()

st.pyplot(fig3)

# ==============================
# TABLEAU
# ==============================
st.subheader(" Statistiques détaillées")

summary = df_filtered.groupby("scenario")["performance"].agg(
    ["mean", "std", "min", "max"]
)

st.dataframe(summary)

# ==============================
# INTERPRÉTATION
# ==============================
st.subheader(" Interprétation automatique")

best_scenario = grouped.idxmax()

interpretation = f"""
- La performance moyenne globale est de **{round(perf_mean, 2)}**.
- L’écart-type de **{round(std_perf, 2)}** permet d’évaluer la stabilité du système.
- Le scénario le plus performant actuellement est **{best_scenario}**.
- Les scénarios intégrant l’IA (tuteur, groupes intelligents, auto-apprentissage) montrent un gain global par rapport au scénario sans assistance.
"""

if p_drop is not None:
    interpretation += f"\n- Le taux moyen de difficulté (p_drop) est de **{round(p_drop, 2)}**."

st.markdown(interpretation)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(" Projet PFE - Simulation Multi-Agents avec IA")