import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utilities import visualize_classifier  # Якщо у вас є така утиліта для візуалізації

# Вхідний файл, який містить дані
input_file = '../data_multivar_nb.txt'

# Завантаження даних з файлу за допомогою pandas
df = pd.read_csv(input_file)

# Перевірка перших кількох рядків даних
print(df.head())

# Розділення на вхідні дані (X) та мітки (y)
X = df.iloc[:, :-1].values  # всі стовпці, окрім останнього
y = df.iloc[:, -1].values  # останній стовпець - мітки

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення класифікатора SVM
classifier_svm = svm.SVC(kernel='linear', C=1)

# Тренування класифікатора
classifier_svm.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = classifier_svm.predict(X_test)

# Обчислення показників якості класифікації
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Виведення результатів
print(f"Accuracy of SVM classifier: {accuracy * 100:.2f}%")
print(f"Precision of SVM classifier: {precision * 100:.2f}%")
print(f"Recall of SVM classifier: {recall * 100:.2f}%")
print(f"F1 Score of SVM classifier: {f1 * 100:.2f}%")

# Візуалізація результатів роботи класифікатора
visualize_classifier(classifier_svm, X_test, y_test)