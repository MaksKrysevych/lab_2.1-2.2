# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier  # Для коректної роботи багатокласової класифікації

# Завантажуємо дані
iris = load_iris()
X = iris.data
y = iris.target
names = iris.target_names

# Розділяємо дані на тренувальну та тестову вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# Список моделей
models = [
    ('LR', OneVsRestClassifier(LogisticRegression(max_iter=200))),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Оцінка моделей за допомогою крос-валідації
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()} ({cv_results.std()})")

# Створення boxplot для візуалізації результатів
plt.boxplot(results, tick_labels=names)  # Замінили 'labels' на 'tick_labels'
plt.title('Оцінка моделей')
plt.show()

# Навчання моделі на тренувальних даних та передбачення на тестових
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінка якості моделі
print(f"\nТочність на тестовому наборі: {accuracy_score(Y_validation, predictions)}")

# Матриця помилок
print("\nМатриця помилок:")
print(confusion_matrix(Y_validation, predictions))

# Звіт про класифікацію
print("\nЗвіт про класифікацію:")
print(classification_report(Y_validation, predictions))

# Прогноз для нової квітки
X_new = np.array([[5, 2.9, 1, 0.2]])  # Нові дані квітки
print(f"\nФорма масиву X_new: {X_new.shape}")

# Прогнозування
prediction = model.predict(X_new)
print(f"Прогноз: {prediction}")
print(f"Спрогнозована мітка: {names[prediction[0]]}")  # Вивести назву класу
