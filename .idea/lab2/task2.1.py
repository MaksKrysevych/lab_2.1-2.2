import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл
input_file = 'income_data.txt'

# Зчитування даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class2 += 1

print("Кількість точок <=50K:", count_class1)
print("Кількість точок >50K:", count_class2)

# Масив numpy
X = np.array(X)
y = np.array(y)

# Кодування ознак
label_encoders = []
X_encoded = np.empty(X.shape, dtype=object)

for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

# Кодування цільової змінної
label_encoder_y = preprocessing.LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Преобразування у числові значення
X = X_encoded.astype(int)

# Масштабування числових атрибутів
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розбиття даних зі збереженням балансу класів
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Класифікатор
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=20000))
classifier.fit(X_train, y_train)

# Прогноз
y_test_pred = classifier.predict(X_test)

# Метрики
print("Accuracy:", round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_test_pred, average='weighted', zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_test_pred, average='weighted') * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test, y_test_pred, average='weighted') * 100, 2), "%")
print("\nClassification report:\n", classification_report(y_test, y_test_pred, target_names=label_encoder_y.classes_, zero_division=0))

# Класифікація окремої точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = []
label_enc_index = 0

for i in range(len(input_data)):
    if input_data[i].isdigit():
        input_data_encoded.append(int(input_data[i]))
    else:
        le = label_encoders[label_enc_index]
        input_data_encoded.append(int(le.transform([input_data[i]])[0]))
        label_enc_index += 1

input_data_encoded = scaler.transform([input_data_encoded])  # Масштабування!
predicted_class = classifier.predict(input_data_encoded)
print("Predicted income class:", label_encoder_y.inverse_transform(predicted_class)[0])
