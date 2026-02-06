import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# df = pd.read_csv("iris.csv")
# print(df.columns)
#


# Mapping dataset name -> path + target column + type
datasets_classification = {
    "Iris": {
        "path": "iris.csv",
        "target": "class",   # <-- corectat
        "type": "classification"
    },
    "Pima Indians Diabetes": {
        "path": "diabetes.csv",
        "target": "Outcome",
        "type": "classification"
    }
}


models_classification = {
    "k-NN (k=3)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]),
    "k-NN (k=10)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=10))]),
    "MLP (10 neurons)": Pipeline([("scaler", StandardScaler()),
                                  ("clf", MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42))]),
    "SVM (RBF kernel)": Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))]),
    "SVM (Linear kernel)": Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(kernel="linear", C=1.0, random_state=42))]),
    "Decision Tree (max_depth=3)": Pipeline(
        [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=3, random_state=42))]),
    "Decision Tree (max_depth=None)": Pipeline(
        [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))])
}


def evaluate_classification_dataset(dataset_info):
    # Load dataset
    df = pd.read_csv(dataset_info["path"])
    X = df.drop(columns=[dataset_info["target"]]).values
    y_raw = df[dataset_info["target"]].values

    # Encode target if needed
    le = None
    if dataset_info["type"] == "classification" and df[dataset_info["target"]].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw

    # cross-validation
    print(f"\n=== Results for {dataset_info['target']} ({dataset_info['path']}) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models_classification.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # arbitrary predictions
    print("\n===  predictions for the arbitrary numbers me gaved ===")
    if dataset_info["target"] == "species":  # Iris example
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    elif dataset_info["target"] == "Outcome":  # Pima example
        sample = np.array([[2, 130, 70, 20, 80, 25.6, 0.5, 33]])
    else:
        sample = X[:1]  # fallback

    for name, model in models_classification.items():
        model.fit(X, y)
        pred = model.predict(sample)
        if le:
            pred = le.inverse_transform(pred)
        print(f"{name} → {pred[0]}")


# run for all datasets
for ds_name, ds_info in datasets_classification.items():
    evaluate_classification_dataset(ds_info)
