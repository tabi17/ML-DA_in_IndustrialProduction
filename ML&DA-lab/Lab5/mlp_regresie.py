import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# hidden_layer_sizes=(50,25) -> două straturi ascunse: 50 și 25 neuroni
# alpha=0.0001 -> regularizare L2 pentru a evita overfitting
# solver='adam' -> optimizator recomandat pentru regresie
mlp_regr = MLPRegressor(hidden_layer_sizes=(50,25), alpha=1e-4, solver='adam', max_iter=1000, random_state=1)

# === 3️⃣ Antrenare model ===
mlp_regr.fit(X_train, y_train)

# === 4️⃣ Evaluare model ===
y_pred = mlp_regr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"MSE pe test set: {mse_test:.2f}")

# === 5️⃣ Cross-validation ===
cv_scores = -cross_val_score(mlp_regr, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE mediu (cross-val): {cv_scores.mean():.2f}, deviație standard: {cv_scores.std():.2f}")

# === 6️⃣ Grafic: valori reale vs prezise ===
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicții MLP")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect fit")
plt.xlabel("Valori reale")
plt.ylabel("Predicții MLP")
plt.title("MLP Regressor: reale vs prezise")
plt.legend()
plt.show()

