import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Завантаження набору даних Iris
iris = load_iris()
X = iris.data  # Чотири ознаки
y = iris.target  # Справжні мітки (не використовуються для кластеризації)

# Ініціалізація та навчання моделі KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeans.fit(X)

# Прогнозування міток кластерів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів (беремо лише перші дві ознаки для простоти)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X')
plt.title("Кластеризація набору Iris (k=3)")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
plt.grid(True)
plt.savefig("iris_kmeans_plot.png")
plt.show()
