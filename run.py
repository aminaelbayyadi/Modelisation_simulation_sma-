from model.simulation_model import SimulationModel
import matplotlib.pyplot as plt
import pandas as pd
import os
import random  

NUM_STUDENTS = 10
NUM_STEPS = 20

# -------- SCÉNARIO 1 : SANS TUTEUR --------
model1 = SimulationModel(NUM_STUDENTS, use_tutor=False, smart_grouping=False)
for _ in range(NUM_STEPS):
    model1.step()
perf1 = model1.performance_history


# -------- SCÉNARIO 2 : AVEC TUTEUR --------
model2 = SimulationModel(NUM_STUDENTS, use_tutor=True, smart_grouping=False)
for _ in range(NUM_STEPS):
    model2.step()
perf2 = model2.performance_history


# -------- SCÉNARIO 3 : GROUPES INTELLIGENTS --------
model3 = SimulationModel(NUM_STUDENTS, use_tutor=True, smart_grouping=True)
for _ in range(NUM_STEPS):
    model3.step()
perf3 = model3.performance_history


# -------- GRAPHIQUE --------
plt.plot(perf1, label="Sans Tuteur")
plt.plot(perf2, label="Avec Tuteur")
plt.plot(perf3, label="Groupes Intelligents")

plt.title("Comparaison des scénarios")
plt.xlabel("Tours")
plt.ylabel("Performance")
plt.legend()
plt.show()


# -------- FONCTION EXPÉRIENCE --------
def run_experiment(num_runs, num_students, num_steps, use_tutor, smart_grouping):

    all_results = []

    for run in range(num_runs):

        model = SimulationModel(num_students, use_tutor, smart_grouping)

        for _ in range(num_steps):
            model.step()

        for student in model.students:

            #  PERFORMANCE AMÉLIORÉE 
            perf = student.competence * student.motivation

            #  effet de la charge
            perf -= 0.1 * student.charge_travail

            #  bruit aléatoire
            perf += random.uniform(-0.1, 0.1)

            #  normalisation
            perf = max(0, min(perf, 1))

            all_results.append({
                "run": run,
                "type": "student",
                "performance": perf,
                "tutor": use_tutor,
                "smart_group": smart_grouping,
                "competence": student.competence,
                "motivation": student.motivation,
                "interactions": student.interactions,
                "charge": student.charge_travail,
                "engagement": student.interactions / (student.charge_travail + 1)
            })

    return all_results


# -------- LANCER EXPÉRIENCES --------
results = []

results += run_experiment(10, 10, 20, False, False)
results += run_experiment(10, 10, 20, True, False)
results += run_experiment(10, 10, 20, True, True)

df = pd.DataFrame(results)


# -------- SAUVEGARDE --------
df.to_csv("data/results.csv", index=False)
print("Données sauvegardées ")


# -------- ENTRAÎNEMENT --------
from ml.ml_model import train_model
ml_model = train_model()


# -------- ÉVALUATION --------
from ml.evaluation import evaluate_model
evaluate_model()