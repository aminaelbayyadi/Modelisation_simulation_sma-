import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():

    # 1. Charger les données
    df = pd.read_csv("data/results.csv")

    # 2. Nettoyage
    df = df.dropna()

    # 3. Définir la cible (étudiant en difficulté)
    df["p_drop"] = df["performance"] < 0.7

    # 4. Feature engineering 
    df["engagement"] = df["interactions"] / (df["charge"] + 1)

    # 5. Variables d'entrée
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

    # 6. Split train/test (pour validation interne)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Modèle optimisé
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # 8. Entraînement
    model.fit(X_train, y_train)

    # 9. Évaluation rapide (debug)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(" Accuracy (train_model) :", accuracy)

    return model