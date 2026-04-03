import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/results.csv")

def scenario_name(row):
    if not row["tutor"] and not row["smart_group"]:
        return "Sans tuteur"
    elif row["tutor"] and not row["smart_group"]:
        return "Avec tuteur"
    else:
        return "Groupes intelligents"

df["scenario"] = df.apply(scenario_name, axis=1)

# moyenne
grouped = df.groupby("scenario")["performance"].mean()

print(grouped)
# écart-type
summary = df.groupby("scenario")["performance"].agg(["mean", "std", "min", "max"])

print("\n=== Tableau des résultats ===")
print(summary)
# graphique
grouped.plot(kind="bar")
# tableau statistique complet

plt.title("Performance moyenne par scénario")
plt.ylabel("Performance")
plt.xticks(rotation=0)

plt.show()
summary.to_csv("data/summary.csv")