"""
Validation des hypotheses.
S1 : entraide CLASSE entiere, sans tuteur
S2 : tuteur (qui utilise un RANDOM FOREST pour cibler ses interventions)
S3 : entraide GROUPE (homogene) + tache adaptee, SANS tuteur
S4 : auto-apprentissage (avec tuteur RF)
H3 : S1 vs S3 (les deux sans tuteur) -> effet pur du groupement.
"""
import os, random, json
import warnings
warnings.filterwarnings("ignore")            # masque les UserWarning sklearn/joblib
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from model.simulation_model import SimulationModel
from analysis.metrics import welch_t_test, cohens_d, anova_test

NUM_RUNS, NUM_STUDENTS, NUM_STEPS = 50, 20, 30
FEATURES = ["tutor", "smart_group", "interactions", "charge", "engagement"]

os.makedirs("data", exist_ok=True)
if os.path.exists("data/results.csv"):
    os.remove("data/results.csv")
random.seed(42); np.random.seed(42)


def run_experiment(use_tutor, smart_grouping, auto_learning=False,
                   peer_help=True, help_scope="classe", grouping_mode="homogene",
                   ml_model=None):
    rows, perf_run = [], []
    for run in range(NUM_RUNS):
        s = 42 + run
        random.seed(s); np.random.seed(s)
        m = SimulationModel(NUM_STUDENTS, use_tutor, smart_grouping, auto_learning,
                            seed=s, peer_help=peer_help, help_scope=help_scope,
                            ml_model=ml_model)
        m.grouping_mode = grouping_mode
        for _ in range(NUM_STEPS):
            m.step()
        interv = getattr(m.tutor, "interventions", 0) if use_tutor else 0
        pr = []
        for st in m.students:
            # Performance = competence x motivation (formule officielle P = C x M).
            # Plus de penalite de charge : 'charge' n'est donc PLUS dans l'etiquette
            # -> on peut l'utiliser comme variable du Random Forest sans data leakage.
            perf = st.competence * st.motivation
            perf += random.uniform(-0.05, 0.05)
            perf = max(0.0, min(perf, 1.0))
            pr.append(perf)
            rows.append({"run": run, "performance": perf, "tutor": use_tutor,
                         "smart_group": smart_grouping, "auto_learning": auto_learning,
                         "competence": st.competence, "motivation": st.motivation,
                         "interactions": st.interactions, "charge": st.charge_travail,
                         "engagement": getattr(st, "actions", 0) / NUM_STEPS,
                         "p_drop": 1 if perf < 0.6 else 0, "interventions": interv})
        perf_run.append(float(np.mean(pr)))
    return rows, np.array(perf_run)


def train_rf(rows):
    d = pd.DataFrame(rows).dropna()
    d = d[d["performance"] > 0]
    y = d["p_drop"].astype(int)
    if y.nunique() < 2:
        return None, None
    X = d[FEATURES].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=120, max_depth=6, class_weight="balanced",
                                random_state=42, n_jobs=1).fit(Xtr, ytr)
    yp = rf.predict(Xte)
    scores = {"accuracy": round(accuracy_score(yte,yp),3),
              "f1": round(f1_score(yte,yp),3), "recall": round(recall_score(yte,yp),3)}
    return rf, scores


# ===== ETAPE 0 : mise en route — on entraine le Random Forest du tuteur =====
print("Pre-entrainement du Random Forest du tuteur...")
boot = []
boot += run_experiment(False, False, help_scope="classe")[0]
boot += run_experiment(True,  False, help_scope="classe")[0]                       # S2 (heuristique)
boot += run_experiment(False, True,  help_scope="groupe")[0]                       # S3
boot += run_experiment(True,  False, auto_learning=True, help_scope="classe")[0]   # S4 (heuristique)
tutor_rf, _ = train_rf(boot)
print("  -> Random Forest du tuteur pret." if tutor_rf else "  -> repli heuristique.")

# ===== ETAPE 1 : les 4 scenarios FINAUX (le tuteur S2/S4 utilise le RF) =====
rows_s1, p_s1 = run_experiment(False, False, help_scope="classe")
rows_s2, p_s2 = run_experiment(True,  False, help_scope="classe", ml_model=tutor_rf)
rows_s3, p_s3 = run_experiment(False, True,  help_scope="groupe", grouping_mode="homogene")
rows_s4, p_s4 = run_experiment(True,  False, auto_learning=True, help_scope="classe", ml_model=tutor_rf)
_, coop  = run_experiment(False, False, peer_help=True,  help_scope="classe")
_, indiv = run_experiment(False, False, peer_help=False, help_scope="classe")

df = pd.DataFrame(rows_s1 + rows_s2 + rows_s3 + rows_s4)
df["scenario"] = df.apply(lambda r: "S4" if r["auto_learning"]
                          else ("S3" if (not r["tutor"]) and r["smart_group"]
                          else ("S2" if r["tutor"] and not r["smart_group"] else "S1")), axis=1)
perf = {"S1": p_s1, "S2": p_s2, "S3": p_s3, "S4": p_s4}

print("\n============== VALIDATION DES HYPOTHESES ==============\n")
t,p = welch_t_test(coop, indiv); d = cohens_d(coop, indiv)
print(f"H1 collaboration : {coop.mean():.4f} vs {indiv.mean():.4f}  p={p:.2e}  d={d:.3f}  -> {'VALIDEE' if coop.mean()>indiv.mean() and p<0.05 else 'NON'}")
t,p = welch_t_test(perf['S2'], perf['S1']); d = cohens_d(perf['S2'], perf['S1'])
print(f"H2 tuteur (RF)   : S2={perf['S2'].mean():.4f} vs S1={perf['S1'].mean():.4f}  p={p:.2e}  d={d:.3f}  -> {'VALIDEE' if perf['S2'].mean()>perf['S1'].mean() and p<0.05 else 'NON'}")
t,p = welch_t_test(perf['S3'], perf['S1']); d = cohens_d(perf['S3'], perf['S1'])
print(f"H3 groupement    : S3={perf['S3'].mean():.4f} vs S1={perf['S1'].mean():.4f}  p={p:.2e}  d={d:.3f}  -> {'VALIDEE' if perf['S3'].mean()>perf['S1'].mean() and p<0.05 else 'NON'}")

# H4 : Random Forest sur les donnees finales
df.to_csv("data/results.csv", index=False)
_, rf_scores = train_rf(df.to_dict("records"))
print(f"H4 Random Forest : acc={rf_scores['accuracy']} F1={rf_scores['f1']} recall={rf_scores['recall']}  -> {'VALIDEE' if rf_scores['recall']>=0.7 else 'A DISCUTER'}")

F,pA = anova_test(perf['S1'],perf['S2'],perf['S3'],perf['S4'])
print(f"ANOVA : F={F:.2f} p={pA:.2e}")
print("\nPerf/scenario :", {k: round(v.mean(),4) for k,v in perf.items()})
print("Interventions tuteur (S2) :", round(np.mean([r['interventions'] for r in rows_s2]),1))

def _p(a,b): return float(stats.ttest_ind(a,b,equal_var=False)[1])
hyp = {
 "H1": {"titre":"La collaboration ameliore la performance","metrique":"Collaboratif vs individuel",
        "valeur":f"{coop.mean():.3f} vs {indiv.mean():.3f}","gain_pct":round((coop.mean()-indiv.mean())/indiv.mean()*100,1),
        "p_value":_p(coop,indiv),"cohens_d":round(cohens_d(coop,indiv),3),"valide":bool(coop.mean()>indiv.mean() and _p(coop,indiv)<0.05)},
 "H2": {"titre":"Le tuteur ameliore la performance","metrique":"S2 vs S1",
        "valeur":f"{perf['S2'].mean():.3f} vs {perf['S1'].mean():.3f}","gain_pct":round((perf['S2'].mean()-perf['S1'].mean())/perf['S1'].mean()*100,1),
        "p_value":_p(perf['S2'],perf['S1']),"cohens_d":round(cohens_d(perf['S2'],perf['S1']),3),"valide":bool(perf['S2'].mean()>perf['S1'].mean() and _p(perf['S2'],perf['S1'])<0.05)},
 "H3": {"titre":"Le groupement intelligent ameliore la performance","metrique":"S3 vs S1 (sans tuteur)",
        "valeur":f"{perf['S3'].mean():.3f} vs {perf['S1'].mean():.3f}","gain_pct":round((perf['S3'].mean()-perf['S1'].mean())/perf['S1'].mean()*100,1),
        "p_value":_p(perf['S3'],perf['S1']),"cohens_d":round(cohens_d(perf['S3'],perf['S1']),3),"valide":bool(perf['S3'].mean()>perf['S1'].mean() and _p(perf['S3'],perf['S1'])<0.05)},
 "H4": {"titre":"Utiliser l'IA pour detecter les apprenants en difficulte","metrique":"Accuracy / F1 / Recall",
        "accuracy":rf_scores["accuracy"],"f1":rf_scores["f1"],"recall":rf_scores["recall"],"valide":bool(rf_scores["recall"]>=0.7)},
 "ANOVA": {"F":round(F,2),"p_value":float(pA),"significatif":bool(pA<0.05)},
}
with open("data/hypotheses.json","w",encoding="utf-8") as f:
    json.dump(hyp,f,ensure_ascii=False,indent=2)
print("\ndata/hypotheses.json exporte.")

# ===== Rapport detaille du Random Forest (H4), affiche UNE seule fois =====
print("\n" + "="*60)
print("  RAPPORT DETAILLE DU RANDOM FOREST (H4)")
print("="*60)
from ml.ml_model import train_model
train_model()