"""
Justification du choix k=3 :
  - Cluster 0 : étudiants en difficulté → soutien fort
  - Cluster 1 : étudiants moyens → soutien modéré
  - Cluster 2 : étudiants autonomes → soutien minimal
Cette tripartition reflète la pratique pédagogique réelle (différenciation
en 3 niveaux). Une analyse par méthode du coude pourrait être ajoutée
en perspective d'amélioration.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


N_CLUSTERS = 3
RANDOM_STATE = 42


def cluster_students(students):
    """
    Regroupe les étudiants en N_CLUSTERS clusters basés sur leurs
    caractéristiques d'apprentissage.

    Returns:
        labels: numpy array de taille len(students) avec un cluster par étudiant.
                Le cluster 0 est conventionnellement le "plus à risque"
                (plus faible compétence × motivation moyennes).
    """
    if len(students) < N_CLUSTERS:
        # Pas assez d'étudiants pour clusteriser → tous dans le même groupe
        return np.zeros(len(students), dtype=int)

    # 1) Construction de la matrice de features
    # Plus de features → clusters plus informatifs
    data = np.array([
        [
            s.competence,
            s.motivation,
            s.charge_travail,
            s.interactions,
        ]
        for s in students
    ], dtype=float)

    # 2) Vérification : si toutes les valeurs sont identiques, KMeans plante
    if data.std(axis=0).sum() == 0:
        return np.zeros(len(students), dtype=int)

    # 3) Standardisation : crucial car charge ∈ [0,5], interactions peut
    # être grand, alors que compétence/motivation ∈ [0,1].
    # Sans standardisation, charge et interactions dominent les distances.
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 4) Clustering
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,           # ← évite le warning sklearn récent
    )
    labels = kmeans.fit_predict(data_scaled)

    # 5) Réordonnancement des clusters par "niveau pédagogique"
    # Cluster 0 = le plus à risque (faible competence × motivation moyenne)
    # Cluster 2 = le plus autonome
    # → permet une interprétation cohérente dans simulation_model.py
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_scores = centroids_original[:, 0] * centroids_original[:, 1]
    # ordre : index du cluster avec score le plus faible → 0,
    # le plus élevé → 2
    order = np.argsort(cluster_scores)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels], dtype=int)

    return labels