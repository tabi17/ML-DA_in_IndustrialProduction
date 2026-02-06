import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


datasets_regression = {
    "Iris": {
        "path": "iris.csv",
        "target": "petallength",
        "type": "regression"
    },
    "Pima Indians Diabetes": {
        "path": "diabetes.csv",
        "target": "Glucose",
        "type": "regression"
    },
    "Cars": {
        "path": "cars.csv",
        "target": "sales",
        "type": "regression"
    }
}


models = {
    "k-NN (k=3)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor(n_neighbors=3))
    ]),
    "k-NN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor(n_neighbors=7))
    ]),
    "MLP (10 neurons)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MLPRegressor(hidden_layer_sizes=(10,), max_iter=2000, random_state=42))
    ]),
    "SVR (RBF kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SVR(kernel="rbf", C=1.0, gamma="scale"))
    ]),
    "SVR (Linear kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SVR(kernel="linear", C=1.0))
    ]),
    "Decision Tree (max_depth=3)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", DecisionTreeRegressor(max_depth=3, random_state=42))
    ]),
    "Decision Tree (max_depth=None)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", DecisionTreeRegressor(random_state=42))
    ])
}




def evaluate_regression_dataset(dataset_info):
    print(f"\n=== Regression results for {dataset_info['path']} (target: {dataset_info['target']}) ===")
    df = pd.read_csv(dataset_info['path'])

    # Handle categorical features (only Iris 'class' as example)
    X = df.drop(columns=[dataset_info['target']])
    y = df[dataset_info['target']]

    # Convert categorical to numeric if exists
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X = X.values
    y = y.values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_dict = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            print(f"{name}: mean R²={scores.mean():.4f}, std={scores.std():.4f}")
            scores_dict[name] = {"mean": scores.mean(), "std": scores.std()}
        except Exception as e:
            print(f"{name} failed: {e}")
            scores_dict[name] = {"mean": np.nan, "std": np.nan}


    # arbitrary prediction
    print("\n=== Arbitrary prediction ===")
    if dataset_info['path'] == "iris.csv":
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # sample arbitrary features
    elif dataset_info['path'] == "diabetes.csv":
        sample = np.array([[2, 130, 70, 20, 80, 25.6, 0.5, 33]])
    elif dataset_info['path'] == "cars.csv":
        sample = np.array([[28, 0, 23, 0, 4099]])

    for name, model in models.items():
        try:
            model.fit(X, y)
            pred = model.predict(sample)[0]
            print(f"{name} → predicted {dataset_info['target']}: {pred:.2f}")
        except Exception as e:
            print(f"{name} prediction failed: {e}")

    # bar chart of R²

    plt.figure(figsize=(10, 5))
    model_names = list(scores_dict.keys())
    mean_scores = [scores_dict[name]['mean'] for name in model_names]
    std_scores = [scores_dict[name]['std'] for name in model_names]
    plt.bar(model_names, mean_scores, yerr=std_scores, alpha=0.7, capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("R²")
    plt.title(f"Regression performance ({dataset_info['path']})")
    plt.tight_layout()
    plt.show()



# run for all datasets

for ds_name, ds_info in datasets_regression.items():
    evaluate_regression_dataset(ds_info)
