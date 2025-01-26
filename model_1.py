"""
Przypadek 1:
- 1 warstwa ukryta (32 neurony)
- Aktywacja: ReLU
- Optymalizator: Adam
- Liczba epok: 20
- batch_size: 32
- EarlyStopping: wyłączone
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD

# Ustawienia losowości (opcjonalne)
np.random.seed(42)
tf.random.set_seed(42)

# 1. Wczytanie danych
data = pd.read_csv('Churn_Modelling.csv')

# 2. Podstawowa analiza danych (opcjonalnie)
data.info()
print(data.isnull().sum())
sns.countplot(x='Exited', data=data)
plt.title('Rozkład klasy docelowej (Churn)')
plt.show()

# 3. Usunięcie niepotrzebnych kolumn
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Kodowanie zmiennych kategorycznych
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# 4. Podział na cechy (X) i etykietę (y)
X = data.drop('Exited', axis=1)
y = data['Exited']

# 5. Podział na zbiory
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Nadpróbkowanie
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

# Standaryzacja
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()

X_train_res[numeric_features] = scaler.fit_transform(X_train_res[numeric_features])
X_val[numeric_features]       = scaler.transform(X_val[numeric_features])
X_test[numeric_features]      = scaler.transform(X_test[numeric_features])

# Budowa modelu: 1 warstwa ukryta (32 neurony), ReLU
regularizer = l2(0.001)
model = Sequential([
    Dense(32, input_dim=X_train_res.shape[1], kernel_regularizer=regularizer, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Kompilacja: Adam, binary_crossentropy
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Trenowanie: 20 epok, batch_size=32, EarlyStopping wyłączone
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Ewaluacja
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Macierz konfuzji')
plt.xlabel('Przewidywane')
plt.ylabel('Prawdziwe')
plt.show()
