import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

# 1. Генерація даних
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# 2. Оцінка ширини вікна
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=300)

# 3. Побудова моделі MeanShift
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift.fit(X)

# 4. Отримання міток і центрів кластерів
labels = meanshift.labels_
cluster_centers = meanshift.cluster_centers_
n_clusters = len(np.unique(labels))

# 5. Візуалізація результатів
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.75, marker='X')
plt.title(f"Кластеризація методом зсуву середнього (Кількість кластерів: {n_clusters})")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.grid(True)
plt.savefig("meanshift_clusters.png")
plt.show()

# 6. Виведення кількості кластерів
print(f"Кількість знайдених кластерів: {n_clusters}")
