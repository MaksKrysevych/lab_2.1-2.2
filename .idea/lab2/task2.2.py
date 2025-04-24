# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Генерація нелінійно розділеного набору даних
X, y = datasets.make_moons(n_samples=300, noise=0.25, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Функція для оцінки класифікатора
def evaluate_model(y_true, y_pred, model_name):
    print(f"=== Результати для {model_name} ===")
    print("Точність (Accuracy):", accuracy_score(y_true, y_pred))
    print("Матриця сплутування (Confusion Matrix):\n", confusion_matrix(y_true, y_pred))
    print("Класифікаційний звіт:\n", classification_report(y_true, y_pred))
    print("-" * 60)

# Поліноміальне ядро
svm_poly = SVC(kernel='poly', degree=8)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
evaluate_model(y_test, y_pred_poly, "Поліноміальне ядро")

# Гаусівське ядро (RBF)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
evaluate_model(y_test, y_pred_rbf, "Гаусівське ядро (RBF)")

# Сигмоїдальне ядро
svm_sigmoid = SVC(kernel='sigmoid')
svm_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = svm_sigmoid.predict(X_test)
evaluate_model(y_test, y_pred_sigmoid, "Сигмоїдальне ядро")

# (Необов'язково) Візуалізація меж прийняття рішень
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=40)
    plt.title(title)
    plt.xlabel("Ознака 1")
    plt.ylabel("Ознака 2")
    plt.tight_layout()
    plt.show()

# Візуалізація результатів
plot_decision_boundary(svm_poly, X, y, "Поліноміальне ядро (degree=8)")
plot_decision_boundary(svm_rbf, X, y, "Гаусівське ядро (RBF)")
plot_decision_boundary(svm_sigmoid, X, y, "Сигмоїдальне ядро")
