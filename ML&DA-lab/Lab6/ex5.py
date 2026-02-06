# random forest classifier xxample for Iris datasets, metric:accuracy
#and
#radnom forest regresion example for iris dataset, target=sepal width
#                           ,input= celelalte atribute, metric: R^2


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# Random Forest CLASSIFICATION

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Classification (Iris)")
print(f"Accuracy: {accuracy:.4f}")


# Random Forest REGRESSION

# Target: sepal width (column 1)
y_reg = iris.data[:, 1]

# Inputs: the other attributes
X_reg = iris.data[:, [0, 2, 3]]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Regression (Iris)")
print(f"R² score (sepal width prediction): {r2:.4f}")
