#2a
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2].reshape(-1, 1)  # Petal length
y = iris.data[:, 3]                 # Petal width

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fit linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict
y_pred = lin_reg.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression on Iris:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

plt.scatter(X_test, y_test, color='blue', label='True')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Iris")
plt.legend()
plt.show()

