import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def evaluate_model():

    # 1. Charger données
    df = pd.read_csv("data/results.csv")

    # 2. Nettoyage
    df = df.dropna()

    # 3. Même logique que ml_model.py (IMPORTANT)
    df["label"] = df["performance"] < 0.7

    # 4. Feature engineering
    df["engagement"] = df["interactions"] / (df["charge"] + 1)

    # 5. Variables
    X = df[[
        "tutor",
        "smart_group",
        "competence",
        "motivation",
        "interactions",
        "charge",
        "engagement"
    ]]

    y = df["label"]

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 7. Modèle
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 8. Prédictions
    y_pred = model.predict(X_test)

    # 9. Métriques
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(" Accuracy :", acc)
    print(" F1-score :", f1)

    print("\n Matrice de confusion :")
    print(cm)

    print("\n Répartition des classes :")
    print(df["label"].value_counts())