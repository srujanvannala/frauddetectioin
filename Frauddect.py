# Fraud Detection Example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# 1. Load sample dataset
# --------------------------
# Example synthetic dataset (you can replace with real data)
# Assume: 'Amount' = transaction amount, 'Time' = transaction time, 'Fraud' = target variable
data = {
    'Time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Amount': [100, 250, 3000, 20, 50, 7000, 10, 500, 2000, 60],
    'Fraud': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# --------------------------
# 2. Split features & target
# --------------------------
X = df[['Time', 'Amount']]
y = df['Fraud']

# --------------------------
# 3. Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------------------------
# 4. Scale features
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 5. Train Logistic Regression Model
# --------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --------------------------
# 6. Predictions
# --------------------------
y_pred = model.predict(X_test_scaled)

# --------------------------
# 7. Evaluation
# --------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
