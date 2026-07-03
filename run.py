import os
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from model.simulation_model import SimulationModel

FEATURES = ["tutor", "smart_group", "interactions", "charge", "engagement"]

# ==============================
# PARAMETRES
# ==============================
NUM_RUNS = 50
NUM_STUDENTS = 20
NUM_STEPS = 30
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ==============================
# DEFINITION DES 4 SCENARIOS (coherente avec run_analysis.py)
#   S1 : entraide CLASSE entiere, sans tuteur
#   S2 : avec tuteur (entraide classe)
#   S3 : entraide GROUPE (homogene par niveau) + tache adaptee, SANS tuteur
#   S4 : auto-apprentissage (avec tuteur)
# Chaque scenario = dict de parametres passes au modele.
# ==============================
SCENARIOS = {
    "S1": dict(use_tutor=False, smart_grouping=False, auto_learning=False,
               help_scope="classe", grouping_mode="homogene"),
    "S2": dict(use_tutor=True,  smart_grouping=False, auto_learning=False,
               help_scope="classe", grouping_mode="homogene"),
    "S3": dict(use_tutor=False, smart_grouping=True,  auto_learning=False,
               help_scope="groupe", grouping_mode="homogene"),
    "S4": dict(use_tutor=True,  smart_grouping=False, auto_learning=True,
               help_scope="classe", grouping_mode="homogene"),
}
LABELS = {
    "S1": "S1 — Sans tuteur (entraide classe)",
    "S2": "S2 — Avec tuteur IA",
    "S3": "S3 — Groupes intelligents (sans tuteur)",
    "S4": "S4 — Auto-apprentissage",
}
COLORS = {"S1": "#2196F3", "S2": "#FF9800", "S3": "#4CAF50", "S4": "#F44336"}


def build_model(params, seed_val, ml_model=None):
    m = SimulationModel(
        NUM_STUDENTS,
        params["use_tutor"],
        params["smart_grouping"],
        params["auto_learning"],
        seed=seed_val,
        help_scope=params["help_scope"],
        ml_model=ml_model if params["use_tutor"] else None,
    )
    m.grouping_mode = params["grouping_mode"]
    return m


def train_tutor_rf():
    """Mise en route : on genere des donnees puis on entraine le Random Forest
    que le tuteur utilisera (memes features que H4, sans data leakage)."""
    rows = []
    for sc, params in SCENARIOS.items():
        for run in range(NUM_RUNS):
            seed_val = 42 + run
            random.seed(seed_val); np.random.seed(seed_val)
            m = build_model(params, seed_val)  # heuristique (pas encore de RF)
            for _ in range(NUM_STEPS):
                m.step()
            for s in m.students:
                perf = max(0.0, min(s.competence * s.motivation + random.uniform(-0.05, 0.05), 1.0))
                rows.append({"tutor": params["use_tutor"], "smart_group": params["smart_grouping"],
                             "interactions": s.interactions, "charge": s.charge_travail,
                             "engagement": getattr(s, "actions", 0) / NUM_STEPS,
                             "p_drop": 1 if perf < 0.6 else 0})
    d = pd.DataFrame(rows)
    y = d["p_drop"].astype(int)
    if y.nunique() < 2:
        return None
    rf = RandomForestClassifier(n_estimators=120, max_depth=6, class_weight="balanced",
                                random_state=42, n_jobs=1)
    rf.fit(d[FEATURES].astype(float), y)
    return rf


print("Pre-entrainement du Random Forest du tuteur...")
TUTOR_RF = train_tutor_rf()


# ==============================
# COURBES D'EVOLUTION
# ==============================
def simulate_scenario(params):
    all_perfs = []
    for run in range(NUM_RUNS):
        seed_val = 42 + run
        random.seed(seed_val); np.random.seed(seed_val)
        m = build_model(params, seed_val, ml_model=TUTOR_RF)
        for _ in range(NUM_STEPS):
            m.step()
        all_perfs.append(m.performance_history)
    return np.mean(all_perfs, axis=0)


print("Calcul des courbes d'evolution...")
courbes = {sc: simulate_scenario(p) for sc, p in SCENARIOS.items()}

print("Performance finale par scenario :")
for sc in ["S1", "S2", "S3", "S4"]:
    print(f"  {LABELS[sc]:42s} : {courbes[sc][-1]:.4f}")

# Graphique d'evolution
plt.figure(figsize=(10, 6))
for sc in ["S1", "S2", "S3", "S4"]:
    plt.plot(courbes[sc], label=LABELS[sc], linewidth=2, color=COLORS[sc])
plt.axhline(0.75, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Seuil reussite")
plt.title(f"Comparaison des scenarios (moyenne sur {NUM_RUNS} runs)")
plt.xlabel("Tours"); plt.ylabel("Performance moyenne")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/evolution_scenarios.png", dpi=120)
plt.show()


# ==============================
# DATASET PAR ETUDIANT (pour le CSV / dashboard / ML)
# ==============================
def run_experiment(scenario_name, params):
    rows = []
    for run in range(NUM_RUNS):
        seed_val = 42 + run
        random.seed(seed_val); np.random.seed(seed_val)
        m = build_model(params, seed_val, ml_model=TUTOR_RF)
        for _ in range(NUM_STEPS):
            m.step()
        total_interventions = getattr(m.tutor, "interventions", 0) if params["use_tutor"] else 0
        for student in m.students:
            # Performance = competence x motivation (formule officielle P = C x M)
            perf = student.competence * student.motivation
            perf += random.uniform(-0.05, 0.05)
            perf = max(0.0, min(perf, 1.0))
            at_risk = 1 if perf < 0.6 else 0
            rows.append({
                "run": run, "scenario": scenario_name, "performance": perf,
                "tutor": params["use_tutor"], "smart_group": params["smart_grouping"],
                "auto_learning": params["auto_learning"],
                "competence": student.competence, "motivation": student.motivation,
                "interactions": student.interactions, "charge": student.charge_travail,
                "engagement": getattr(student, "actions", 0) / NUM_STEPS,
                "at_risk": at_risk, "p_drop": at_risk,
                "interventions": total_interventions,
            })
    return rows


# RESET dataset
if os.path.exists("data/results.csv"):
    os.remove("data/results.csv")

print("\nGeneration du dataset complet...")
results = []
for sc, p in SCENARIOS.items():
    results += run_experiment(sc, p)

df = pd.DataFrame(results)
df.to_csv("data/results.csv", index=False)
print(f"Donnees sauvegardees : {len(df)} lignes")

# Resume par scenario (l'etiquette vient directement de la colonne 'scenario')
print("\nResume par scenario :")
summary = df.groupby("scenario").agg({
    "performance": ["mean", "std"],
    "engagement": "mean",
    "at_risk": "mean",
    "interventions": "mean",
}).round(3)
print(summary)

print("\nPour valider les hypotheses : lance `python run_analysis.py`")
print("Pour le tableau de bord       : lance `streamlit run dashboard.py`")