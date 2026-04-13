from sklearn.cluster import KMeans
import numpy as np

def cluster_students(students):
    data = []

    for s in students:
        data.append([s.competence, s.motivation])

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data)

    return labels