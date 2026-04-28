import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

def evaluate_model():

    # ==============================
    # 1. Charger données
    # ==============================
    df = pd.read_csv("data/results.csv")
    df = df.dropna()

    #  nettoyer
    df = df[df["performance"] > 0]

    #  feature
    df["engagement"] = df["interactions"] / (df["charge"] + 1)

    # ==============================
    # 2. Variables
    # ==============================
    X = df[[
        "tutor",
        "smart_group",
        "competence",
        "motivation",
        "interactions",
        "charge",
        "engagement"
    ]]

    y = df["p_drop"]  

    print("\n Répartition des classes :")
    print(y.value_counts())

    # sécurité
    if y.value_counts().min() < 2:
        print(" Dataset trop petit ou déséquilibré")
        return

    # ==============================
    # 3. Split
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # ==============================
    # 4. Charger modèle
    # ==============================
    model = joblib.load("ml/model.pkl")

    # ==============================
    # 5. Prédiction
    # ==============================
    y_pred = model.predict(X_test)

    # ==============================
    # 6. Métriques
    # ==============================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n✔ Accuracy :", round(acc, 3))
    print("✔ F1-score :", round(f1, 3))

    print("\n Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))