import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


iris = pd.read_csv("iris.csv")
pima = pd.read_csv("diabetes.csv")


X_iris = iris.iloc[:, :-1].values
y_iris_raw = iris.iloc[:, -1]


le = LabelEncoder()
y_iris = le.fit_transform(y_iris_raw)


X_pima = pima.iloc[:, :-1].values
y_pima = pima.iloc[:, -1].values


models = {
    "k-NN (k=3)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=3))
    ]),
    "k-NN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "MLP (10 neurons)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42))
    ]),
    "SVM (RBF kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]),
    "SVM (Linear kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0, random_state=42))
    ]),
    "Decision Tree (max_depth=3)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=42))
    ]),
    "Decision Tree (max_depth=None)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
}


def evaluate_models(X, y, dataset_name):
    print(f"\n=== Rezultate pentru {dataset_name} ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")


evaluate_models(X_iris, y_iris, "Iris")
evaluate_models(X_pima, y_pima, "Pima Indians Diabetes")


print("\n=== Predicții pentru valori arbitrare ===")


sample_iris = np.array([[5.1, 3.5, 1.4, 0.2]])


sample_pima = np.array([[2, 130, 70, 20, 80, 25.6, 0.5, 33]])

for name, model in models.items():
    # Iris
    model.fit(X_iris, y_iris)
    pred_iris = le.inverse_transform(model.predict(sample_iris))[0]
    # Pima
    model.fit(X_pima, y_pima)
    pred_pima = model.predict(sample_pima)[0]

    print(f"{name}: Iris → {pred_iris}, Pima → {'Diabetic' if pred_pima==1 else 'Non-diabetic'}")
