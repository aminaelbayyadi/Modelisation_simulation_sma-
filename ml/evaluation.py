import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


# Mêmes features que ml_model.py (sans data leakage)
FEATURES = [
    "tutor",
    "smart_group",
    "interactions",
    "charge",
    "engagement",
]


def evaluate_model():
    """
    Évalue le modèle Random Forest sur un jeu de test indépendant
    (différent de celui utilisé à l'entraînement).
    """
    # ----- 1. Charger les données -----
    df = pd.read_csv("data/results.csv")
    df = df.dropna()
    df = df[df["performance"] > 0]

    df = df.copy()
    df["engagement"] = df["interactions"] / (df["charge"] + 1)

    y = df["p_drop"]

    print("\nRépartition des classes :")
    print(y.value_counts())
    print(f"Proportion étudiants à risque : {y.mean()*100:.1f}%")

    if y.value_counts().min() < 2:
        print("Dataset trop petit ou déséquilibré.")
        return

    # ----- 2. Split indépendant (random_state différent du training) -----
    _, df_test = train_test_split(
        df,
        test_size=0.3,
        random_state=99,           # ← différent du 42 utilisé à l'entraînement
        stratify=y,
    )
    y_test = df_test["p_drop"]
    X_test = df_test[FEATURES]

    print(f"\nTaille du jeu de test : {len(df_test)} échantillons")

    # ----- 3. Charger et évaluer le modèle -----
    model_path = "ml/model.pkl"
    if not os.path.exists(model_path):
        print(f"\n[Info] {model_path} absent — entraîne d'abord avec train_model().")
        return

    model = joblib.load(model_path)

    print(f"\n{'='*60}")
    print(f"  ÉVALUATION DU MODÈLE RANDOM FOREST")
    print(f"{'='*60}")

    y_pred = model.predict(X_test)

    # Métriques globales
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f" Accuracy : {acc:.3f}")
    print(f" F1-score : {f1:.3f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion :")
    print(f"                Prédit:0   Prédit:1")
    print(f"  Réel:0       {cm[0][0]:>8}   {cm[0][1]:>8}")
    print(f"  Réel:1       {cm[1][0]:>8}   {cm[1][1]:>8}")

    # Rapport par classe
    print(f"\nRapport détaillé :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Non à risque (0)", "À risque (1)"],
        digits=3,
    ))

    # Recall pour la classe positive
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    if len(recall) >= 2:
        print(f"Recall classe 'À risque' : {recall[1]:.3f}")
        print(f"  → {cm[1][1]}/{cm[1][0]+cm[1][1]} étudiants à risque détectés")

    return {
        "accuracy": acc,
        "f1": f1,
        "recall_at_risk": recall[1] if len(recall) >= 2 else 0.0,
        "confusion_matrix": cm.tolist(),
    }