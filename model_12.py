"""
Przykład 12:
- 3 warstwy (8, 4, 2) - bardzo małe warstwy
- Aktywacja: softplus
- Optymalizator: SGD (z momentum=0.9)
- Liczba epok: 50
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
from tensorflow.keras.optimizers import SGD

# Ustawienia losowości (opcjonalne)
np.random.seed(42)
tf.random.set_seed(42)

# 1. Wczytanie danych
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

X = data.drop('Exited', axis=1)
y = data['Exited']

# 2. Podział na zbiory
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 3. Nadpróbkowanie
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

# 4. Standaryzacja
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()

X_train_res[numeric_features] = scaler.fit_transform(X_train_res[numeric_features])
X_val[numeric_features]       = scaler.transform(X_val[numeric_features])
X_test[numeric_features]      = scaler.transform(X_test[numeric_features])

# 5. Budowa modelu z małą liczbą neuronów
model = Sequential([
    Dense(8,  input_dim=X_train_res.shape[1], kernel_regularizer=l2(0.001), activation='softplus'),
    Dropout(0.5),
    Dense(4,  kernel_regularizer=l2(0.001), activation='softplus'),
    Dropout(0.5),
    Dense(2,  kernel_regularizer=l2(0.001), activation='softplus'),
    Dropout(0.5),
    Dense(1,  activation='sigmoid')
])

# 6. Kompilacja z optymalizatorem SGD
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    optimizer=sgd_optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 7. Trenowanie (BEZ EarlyStopping)
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8. Ewaluacja
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
