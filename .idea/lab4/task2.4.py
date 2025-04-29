import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs

# 1. Генерація даних
X, _ = make_blobs(n_samples=100, centers=5, cluster_std=0.80, random_state=42)

# 2. Побудова моделі поширення подібності
affprop = AffinityPropagation(random_state=0)
affprop.fit(X)

# 3. Отримання кластерів
cluster_centers_indices = affprop.cluster_centers_indices_
labels = affprop.labels_
n_clusters = len(cluster_centers_indices)

# 4. Візуалізація результатів
plt.figure(figsize=(6, 6))
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    cluster_members = (labels == i)
    plt.scatter(X[cluster_members, 0], X[cluster_members, 1], s=50, color=colors[i], label=f"Кластер {i+1}")
    plt.scatter(X[cluster_centers_indices[i], 0], X[cluster_centers_indices[i], 1],
                s=200, marker='X', c='black', edgecolors='white')

plt.title(f"Affinity Propagation: {n_clusters} кластерів")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.grid(True)
plt.savefig("affinity_propagation_clusters.png")
plt.show()

# 5. Вивід кількості кластерів
print(f"Кількість знайдених кластерів: {n_clusters}")
