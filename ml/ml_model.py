"""
Module d'entraînement du modèle Random Forest (version simplifiée).

Note méthodologique : ce modèle utilise uniquement des features
indirectes (tutor, smart_group, interactions, charge, engagement)
afin d'éviter le data leakage causé par compétence et motivation
qui interviennent dans la formule de p_drop.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# Features sans data leakage (compétence et motivation exclues car
# elles servent à calculer p_drop)
FEATURES = [
    "tutor",
    "smart_group",
    "interactions",
    "charge",
    "engagement",
]


def train_model():
    """
    Entraîne un Random Forest pour prédire le risque de décrochage,
    en utilisant uniquement des features indirectes (pas de data leakage).
    """
    # ----- 1. Charger les données -----
    df = pd.read_csv("data/results.csv")
    df = df.dropna()
    df = df[df["performance"] > 0]

    if len(df) < 50:
        print("Pas assez de données pour entraîner le modèle.")
        return None

    # ----- 2. Cible -----
    y = df["p_drop"]
    print("\nRépartition des classes (cible p_drop) :")
    print(y.value_counts())
    print(f"Proportion classe minoritaire : {y.mean()*100:.1f}%")

    if y.nunique() < 2:
        print("Une seule classe présente — impossible d'entraîner.")
        return None

    # ----- 3. Vérification des features -----
    for col in FEATURES:
        if col not in df.columns:
            print(f"Colonne manquante : {col}")
            return None

    X = df[FEATURES]

    # ----- 4. Entraînement -----
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT — MODÈLE RANDOM FOREST")
    print(f"{'='*60}")
    print(f"Features utilisées ({len(FEATURES)}) : {FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ----- 5. Métriques globales -----
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n Accuracy : {acc:.3f}")
    print(f" F1-score : {f1:.3f}")

    # ----- 6. Matrice de confusion -----
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion :")
    print(f"                Prédit:0   Prédit:1")
    print(f"  Réel:0       {cm[0][0]:>8}   {cm[0][1]:>8}")
    print(f"  Réel:1       {cm[1][0]:>8}   {cm[1][1]:>8}")

    # ----- 7. Rapport par classe -----
    print(f"\nRapport par classe :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Non à risque (0)", "À risque (1)"],
        digits=3,
    ))

    # ----- 8. Validation croisée -----
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=1)
    print(f"F1-score en cross-validation 5-fold :")
    print(f"   Moyenne : {cv_scores.mean():.3f} | "
          f"Écart-type : {cv_scores.std():.3f}")

    # ----- 9. Feature importance -----
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)
    print(f"\nFeature importance :")
    for name, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"   {name:<15} {imp:.3f}  {bar}")

    # ----- 10. Sauvegarde -----
    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, "ml/model.pkl")
    print(f"\nModèle sauvegardé : ml/model.pkl")

    return model