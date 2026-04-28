import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib


def train_model():

    # ==============================
    # 1. Charger données
    # ==============================
    df = pd.read_csv("data/results.csv")
    df = df.dropna()

    #  nettoyer
    df = df[df["performance"] > 0]

    if len(df) < 50:
        print(" Pas assez de données")
        return None

    # ==============================
    # 2. Feature engineering
    # ==============================
    df["engagement"] = df["interactions"] / (df["charge"] + 1)

    #  bruit réaliste (léger)
    df["competence"] += np.random.normal(0, 0.03, len(df))
    df["motivation"] += np.random.normal(0, 0.03, len(df))

    # ==============================
    # 3. TARGET (CORRECT)
    # ==============================
    y = df["p_drop"]  

    print("\n Répartition classes :")
    print(y.value_counts())

    if y.nunique() < 2:
        print(" Dataset déséquilibré")
        return None

    # ==============================
    # 4. Features
    # ==============================
    features = [
        "tutor",
        "smart_group",
        "competence",
        "motivation",
        "interactions",
        "charge",
        "engagement"
    ]

    for col in features:
        if col not in df.columns:
            print(f" Colonne manquante : {col}")
            return None

    X = df[features]

    # ==============================
    # 5. Split
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ==============================
    # 6. Modèle
    # ==============================
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    # ==============================
    # 7. Entraînement
    # ==============================
    model.fit(X_train, y_train)

    # ==============================
    # 8. Évaluation
    # ==============================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n✔ Accuracy :", round(acc, 3))
    print("✔ F1-score :", round(f1, 3))

    print("\n Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # ==============================
    # 9. Sauvegarde
    # ==============================
    joblib.dump(model, "ml/model.pkl")
    print(" Modèle sauvegardé")

    return model