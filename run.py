import os
from model.simulation_model import SimulationModel
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

NUM_RUNS = 50
NUM_STUDENTS = 20
NUM_STEPS = 30

# ==============================
# FIXER LE HASARD GLOBAL
# ==============================
random.seed(42)
np.random.seed(42)


# ==============================
# SIMULATION MOYENNE
# ==============================
def simulate_scenario(use_tutor, smart_grouping, auto_learning):

    all_perfs = []

    for run in range(NUM_RUNS):

        # seed reproductible mais différent
        random.seed(42 + run)
        np.random.seed(42 + run)

        model = SimulationModel(
            NUM_STUDENTS,
            use_tutor,
            smart_grouping,
            auto_learning
        )

        for _ in range(NUM_STEPS):
            model.step()

        all_perfs.append(model.performance_history)

    return np.mean(all_perfs, axis=0)


# ==============================
# CALCUL SCÉNARIOS
# ==============================
perf1 = simulate_scenario(False, False, False)   # S1
perf2 = simulate_scenario(True, False, False)    # S2
perf3 = simulate_scenario(True, True, False)     # S3
perf4 = simulate_scenario(True, False, True)     # S4

# DEBUG 
print("Final S1:", perf1[-1])
print("Final S2:", perf2[-1])
print("Final S3:", perf3[-1])
print("Final S4:", perf4[-1])


# ==============================
# GRAPHIQUE PROPRE
# ==============================
plt.figure(figsize=(10,6))

plt.plot(perf1, label="Sans Tuteur", linewidth=2)
plt.plot(perf2, label="Avec Tuteur IA", linewidth=3)  # 🔥 important
plt.plot(perf3, label="Groupes Intelligents", linewidth=4)
plt.plot(perf4, label="Auto-apprentissage adaptatif", linewidth=3)

plt.title("Comparaison des scénarios (moyenne sur 50 runs)")
plt.xlabel("Tours")
plt.ylabel("Performance")
plt.legend()
plt.grid()

plt.show()


# ==============================
# EXPÉRIENCE (CSV)
# ==============================
def run_experiment(num_runs, num_students, num_steps,
                   use_tutor, smart_grouping, auto_learning=False):

    all_results = []

    for run in range(num_runs):

        random.seed(100 + run)
        np.random.seed(100 + run)

        model = SimulationModel(
            num_students,
            use_tutor,
            smart_grouping,
            auto_learning
        )

        for _ in range(num_steps):
            model.step()

        # interventions
        interventions = getattr(model.tutor, "interventions", 0) if use_tutor else 0

        for student in model.students:

            #  performance réaliste
            perf = student.competence * student.motivation
            perf -= 0.03 * student.charge_travail
            perf += random.uniform(-0.05, 0.05)

            perf = max(0, min(perf, 1))

            #  p_drop DYNAMIQUE (IMPORTANT)
            # seuil basé sur distribution
            # seuil FIXE stable
            threshold = 0.6
            p_drop = 1 if perf < threshold else 0

            all_results.append({
                "run": run,
                "performance": perf,
                "tutor": use_tutor,
                "smart_group": smart_grouping,
                "auto_learning": auto_learning,
                "competence": student.competence,
                "motivation": student.motivation,
                "interactions": student.interactions,
                "charge": student.charge_travail,
                "engagement": student.interactions / (student.charge_travail + 1),
                "p_drop": p_drop,
                "interventions": interventions / (num_steps * len(model.students))
            })

    return all_results


# ==============================
# RESET DATASET
# ==============================
if os.path.exists("data/results.csv"):
    os.remove("data/results.csv")


# ==============================
# LANCER EXPÉRIENCES
# ==============================
results = []

results += run_experiment(50, 20, 30, False, False, False)
results += run_experiment(50, 20, 30, True, False, False)
results += run_experiment(50, 20, 30, True, True, False)
results += run_experiment(50, 20, 30, True, False, True)

df = pd.DataFrame(results)

df.to_csv("data/results.csv", index=False)
print("Données sauvegardées ✔")


# ==============================
# MACHINE LEARNING
# ==============================
from ml.ml_model import train_model
ml_model = train_model()

from ml.evaluation import evaluate_model
evaluate_model()