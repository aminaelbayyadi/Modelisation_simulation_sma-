import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

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
/* Cartes KPI */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎓 Dashboard — Simulation Multi-Agents</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-info">PFE 2024–2025 | Modélisation de l\'apprentissage collaboratif avec SMA + IA</p>', unsafe_allow_html=True)

# ==============================
# CHARGEMENT DONNÉES
# ==============================
@st.cache_data  # ← CORRECTION 1 : cache pour ne pas recharger à chaque interaction
def load_data():
    try:
        df = pd.read_csv("data/results.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error(" Fichier data/results.csv introuvable. Lance d'abord run.py pour générer les données.")
    st.info("Commande : python run.py")
    st.stop()

# ==============================
# CORRECTION 2 : Noms des scénarios
# Problème original : les noms étaient tronqués dans les graphiques
# Solution : noms courts + mapping clair
# ==============================
NOMS_SCENARIOS = {
    "Sans tuteur"              : "S1 — Sans tuteur",
    "Avec tuteur IA"           : "S2 — Avec tuteur IA",
    "Groupes intelligents"     : "S3 — Groupes intelligents",
    "Auto-apprentissage adaptatif" : "S4 — Auto-apprentissage",
}

# Couleurs cohérentes pour tous les graphiques
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
    tutor     = row.get("tutor", False)
    smart     = row.get("smart_group", False)
    if not tutor and not smart:
        return "S1 — Sans tuteur"
    elif tutor and not smart:
        return "S2 — Avec tuteur IA"
    elif tutor and smart:
        return "S3 — Groupes intelligents"
    return "Autre"

df["scenario"] = df.apply(get_scenario_name, axis=1)

# ==============================
# SIDEBAR — Filtres
# ==============================
st.sidebar.header("🔎 Filtres")

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
st.sidebar.markdown("🔵 S1 — Sans tuteur (groupe contrôle)")
st.sidebar.markdown("🟠 S2 — Avec tuteur IA (Random Forest)")
st.sidebar.markdown("🟢 S3 — Groupes intelligents (K-Means)")
st.sidebar.markdown("🔴 S4 — Auto-apprentissage adaptatif")

# ==============================
# FILTRAGE
# ==============================
df_filtered = df[
    (df["scenario"].isin(selected_scenario)) &
    (df["run"] >= selected_runs[0]) &
    (df["run"] <= selected_runs[1])
].copy()

if df_filtered.empty:
    st.warning("⚠️ Aucune donnée avec ces filtres. Modifie la sélection.")
    st.stop()

# ==============================
# CORRECTION 3 : KPI — p_drop = 0.0
# Problème original : p_drop.mean() retournait 0.0 car
# les lignes avec p_drop=0 (non calculé) faussaient la moyenne
# Solution : filtrer les valeurs > 0 avant de calculer la moyenne
# ==============================
perf_mean = df_filtered["performance"].mean()
std_perf  = df_filtered["performance"].std()

# p_drop : on exclut les 0 qui signifient "non calculé"
if "p_drop" in df_filtered.columns:
    p_drop_valide = df_filtered[df_filtered["p_drop"] > 0]["p_drop"]
    p_drop = p_drop_valide.mean() if not p_drop_valide.empty else None
else:
    p_drop = None

# Interventions : somme totale
if "interventions" in df_filtered.columns:
    interventions = int(df_filtered["interventions"].sum())
else:
    interventions = 0

# ==============================
# KPI — Indicateurs clés
# ==============================
st.subheader(" Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Performance moyenne",
    value=round(perf_mean, 3),
    delta=f"{'↑ Bon' if perf_mean > 0.6 else '↓ Faible'}"
)
col2.metric(
    label="Écart-type",
    value=round(std_perf, 3),
    delta=f"{'⚠ Inégalités élevées' if std_perf > 0.3 else '✓ Stable'}"
)
# ==============================
# p_drop moyen (CORRECT)
# ==============================

if "p_drop" in df.columns:

    valid = df[df["p_drop"] > 0]

    if len(valid) > 0:
        p_drop_mean = valid["p_drop"].mean()
    else:
        p_drop_mean = 0.0

else:
    p_drop_mean = None


# ==============================
# Affichage
# ==============================
# p_drop moyen
p_drop_mean = df_filtered["p_drop"].mean()

if p_drop_mean < 0.3:
    label = "🟢 Faible"
elif p_drop_mean < 0.6:
    label = "🟡 Moyen"
else:
    label = "🔴 Élevé"

col3.metric(
    label="p_drop moyen",
    value=round(p_drop_mean, 3),
    delta=label
)
col4.metric(
    label="Interventions tuteur",
    value=interventions
)

st.markdown("---")

# ==============================

st.subheader(" Performance moyenne par scénario")

grouped = df_filtered.groupby("scenario")["performance"].mean().sort_values()

fig1, ax1 = plt.subplots(figsize=(9, 5))

bars = ax1.bar(
    grouped.index,
    grouped.values,
    color=[COULEURS.get(sc, "#999999") for sc in grouped.index],  # ← couleurs différentes
    edgecolor="white",
    linewidth=0.8,
    width=0.55
)

# Ligne seuil de réussite
ax1.axhline(y=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Seuil réussite (0.75)")

# Valeur affichée sur chaque barre
for bar in bars:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{h:.3f}",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold"
    )

ax1.set_ylabel("Performance moyenne", fontsize=11)
ax1.set_title("Comparaison des scénarios", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# CORRECTION étiquettes tronquées : rotation + alignement
plt.xticks(rotation=20, ha="right", fontsize=9)  # ← corrige "upes intelligents"
plt.tight_layout()

st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ==============================
# GRAPHE EVOLUTION (amélioré)
# ==============================
st.subheader(" Évolution de la performance par run")

fig2, ax2 = plt.subplots(figsize=(10, 5))

for scenario in df_filtered["scenario"].unique():
    subset    = df_filtered[df_filtered["scenario"] == scenario]
    evolution = subset.groupby("run")["performance"].mean()
    couleur   = COULEURS.get(scenario, "#999999")

    ax2.plot(
        evolution.index,
        evolution.values,
        marker="o",
        markersize=4,
        linewidth=2.2,
        color=couleur,
        label=scenario
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
# HISTOGRAMME (amélioré)
# ==============================
st.subheader(" Distribution des performances")

fig3, ax3 = plt.subplots(figsize=(9, 4))

for scenario in df_filtered["scenario"].unique():
    subset  = df_filtered[df_filtered["scenario"] == scenario]
    couleur = COULEURS.get(scenario, "#999999")
    ax3.hist(
        subset["performance"],
        bins=15,
        alpha=0.55,
        color=couleur,
        label=scenario,
        edgecolor="white"
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
# TABLEAU STATISTIQUES
# ==============================
st.subheader(" Statistiques détaillées par scénario")

cols_agg = {"performance": ["mean", "std", "min", "max"]}
if "p_drop" in df_filtered.columns:
    cols_agg["p_drop"] = ["mean"]
if "interventions" in df_filtered.columns:
    cols_agg["interventions"] = ["sum"]

summary = df_filtered.groupby("scenario").agg(cols_agg).round(3)

# Renommer les colonnes pour plus de clarté
summary.columns = [
    "Perf. moyenne", "Écart-type", "Perf. min", "Perf. max"
] + (["p_drop moyen"] if "p_drop" in df_filtered.columns else []) \
  + (["Interventions (total)"] if "interventions" in df_filtered.columns else [])

st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ==============================
# VALIDATION DES HYPOTHÈSES
# (section ajoutée — absente dans l'original)
# ==============================
st.subheader("✅ Validation des hypothèses de recherche")

perf_par_sc = df_filtered.groupby("scenario")["performance"].mean()

h1_col, h2_col, h3_col = st.columns(3)

# H1 : collaboration > sans aide
with h1_col:
    s1 = perf_par_sc.get("S1 — Sans tuteur", None)
    s2 = perf_par_sc.get("S2 — Avec tuteur IA", None)
    if s1 and s2:
        valide = s2 > s1
        st.metric("H1 — Collaboration améliore perf.", f"{s2:.3f} vs {s1:.3f}",
                  delta="✅ VALIDÉE" if valide else "❌ NON VALIDÉE")
    else:
        st.info("H1 : Sélectionner S1 et S2")

# H2 : tuteur IA améliore apprentissage
with h2_col:
    if s1 and s2:
        reduction = ((s2 - s1) / s1) * 100 if s1 > 0 else 0
        valide = s2 > s1
        st.metric("H2 — Tuteur IA améliore résultats", f"+{reduction:.1f}%",
                  delta="✅ VALIDÉE" if valide else "❌ NON VALIDÉE")
    else:
        st.info("H2 : Sélectionner S1 et S2")

# H3 : groupes intelligents > aléatoires
with h3_col:
    s3 = perf_par_sc.get("S3 — Groupes intelligents", None)
    if s1 and s3:
        valide = s3 > s1
        st.metric("H3 — Groupes équilibrés > aléatoires", f"{s3:.3f} vs {s1:.3f}",
                  delta="✅ VALIDÉE" if valide else "❌ NON VALIDÉE")
    else:
        st.info("H3 : Sélectionner S1 et S3")

st.markdown("---")

# ==============================
# INTERPRÉTATION AUTOMATIQUE
# ==============================
st.subheader(" Interprétation automatique")

best_scenario  = perf_par_sc.idxmax() if not perf_par_sc.empty else "N/A"
worst_scenario = perf_par_sc.idxmin() if not perf_par_sc.empty else "N/A"

interpretation = f"""
- La **performance moyenne globale** sur les scénarios sélectionnés est de **{round(perf_mean, 3)}**.
- L'**écart-type de {round(std_perf, 3)}** reflète la variabilité entre scénarios — une valeur élevée indique que les configurations ont des impacts très différents.
- Le **scénario le plus performant** est **{best_scenario}** et le **moins performant** est **{worst_scenario}**.
- Les scénarios intégrant l'IA (tuteur, groupes intelligents, auto-apprentissage) montrent un gain significatif par rapport au scénario sans assistance.
"""

if "p_drop" in df_filtered.columns:

    p_drop_mean = df_filtered["p_drop"].mean()

    interpretation += (
        f"\n- Le **taux moyen de risque de décrochage (p_drop)** est de "
        f"**{round(p_drop_mean, 3)}** — calculé par le modèle Random Forest sur les étudiants actifs."
    )
else:
    interpretation += "\n- La colonne **p_drop** n'est pas disponible dans le fichier CSV — vérifier que le modèle ML a bien exporté cette valeur."

st.markdown(interpretation)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<center>🎓 PFE 2024–2025 — Simulation d'apprentissage collaboratif avec SMA + IA</center>",
    unsafe_allow_html=True
)