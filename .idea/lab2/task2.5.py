import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO  # Needed for plot
import seaborn as sns
import matplotlib.pyplot as plt

# Завантажуємо набір даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# Розділяємо дані на тренувальні та тестові
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# Ініціалізація класифікатора Ridge
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)

# Прогнозування на тестових даних
ypred = clf.predict(Xtest)

# Обчислення показників якості
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))

print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

# Показуємо звіт по класифікації
print('\n\tClassification Report:\n', metrics.classification_report(ytest, ypred))

# Будуємо матрицю плутанини
mat = confusion_matrix(ytest, ypred)

# Візуалізація матриці плутанини через теплову карту
sns.set()
plt.figure(figsize=(8, 6))  # Задаємо розмір графіка
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')

# Зберігаємо графік у форматі SVG
f = BytesIO()
plt.savefig(f, format="svg")
plt.show()
