import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample

# Завантажуємо дані
data = pd.read_csv('income_data.txt', header=None)

# Оголошуємо колонки
data.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status',
                'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss',
                'Hours-per-week', 'Native-country', 'Income']

# Перетворюємо категоріальні змінні на числові за допомогою LabelEncoder
label_encoders = {}
categorical_columns = ['Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship',
                       'Race', 'Sex', 'Native-country', 'Income']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Розділяємо дані на ознаки та мітки
X = data.drop('Income', axis=1)
y = data['Income']

# Нормалізуємо числові ознаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Балансування класів за допомогою upsampling (для дисбалансу класів)
X_resampled, y_resampled = resample(X_scaled, y, replace=True, n_samples=len(y), random_state=42)

# Використовуємо крос-валідацію для оцінки моделей з 3 фолдами
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(class_weight='balanced')
}

# Крос-валідація для кожної моделі
for name, model in models.items():
    try:
        cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=3, error_score='raise')  # Використовуємо 3-фолд крос-валідацію
        print(f"{name} - Accuracy: {cv_scores.mean()} ± {cv_scores.std()}")
    except Exception as e:
        print(f"Error with {name}: {str(e)}")

# Якщо потрібно, можна побудувати звіт по класифікації для кращої моделі, наприклад:
best_model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Приклад, для кращої моделі, змінюйте за потреби
best_model.fit(X_resampled, y_resampled)
y_pred_best = best_model.predict(X_resampled)

print("\nBest Model (Logistic Regression) Classification Report:")
print(classification_report(y_resampled, y_pred_best))
