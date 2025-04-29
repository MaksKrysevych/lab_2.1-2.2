import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Генерація штучних даних
X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# 2. Візуалізація вхідних даних
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Вхідні дані")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.grid(True)
plt.savefig("input_data_plot.png")
plt.show()

# 3. Побудова моделі KMeans
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X)

# 4. Прогнозування міток
y_kmeans = kmeans.predict(X)

# 5. Відображення результатів кластеризації
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X')
plt.title("Кластеризація методом K-середніх")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.grid(True)
plt.savefig("clustered_data_plot.png")
plt.show()
