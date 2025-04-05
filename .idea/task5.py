import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utilities import visualize_classifier

# Вхідний файл
input_file = 'data_multivar_nb1.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# ------------------------------
# РОЗБИВКА НА ТРЕНУВАЛЬНІ І ТЕСТОВІ ДАНІ
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення і навчання класифікатора
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Прогнозування
y_test_pred = classifier.predict(X_test)

# Оцінка якості
accuracy = 100.0 * accuracy_score(y_test, y_test_pred)
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Візуалізація
visualize_classifier(classifier, X_test, y_test)

# ------------------------------
# ПЕРЕХРЕСНА ПЕРЕВІРКА (CROSS-VALIDATION)
# ------------------------------
num_folds = 3

accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Cross-validated Accuracy: ", round(100 * accuracy_values.mean(), 2), "%")

precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: ", round(100 * precision_values.mean(), 2), "%")

recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: ", round(100 * recall_values.mean(), 2), "%")

f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1 score: ", round(100 * f1_values.mean(), 2), "%")
