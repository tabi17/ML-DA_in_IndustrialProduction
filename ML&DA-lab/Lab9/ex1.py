import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    return tn / (tn + fp)


def classification_metrics(y_true, y_pred, y_prob=None, label="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    spec = specificity_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    print(f"\n{label}  -")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"AUC: {auc:.4f}")

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=label)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()




datasets_dict = {
    "Iris": load_iris(),
    "Pima": pd.read_csv("/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/Lab8/diabetes.csv")  # CSV with column 'Outcome' as target
}


# Loop over datasets
for dataset_name, data in datasets_dict.items():
    print(f"\n Dataset: {dataset_name}...................................")

    if dataset_name == "Iris":
        X = data.data
        y = data.target
        # For binary metrics, select only class 0 vs 1
        X = X[y != 2]
        y = y[y != 2]
    else:  # Pima
        df = data
        y = df['Outcome'].values
        X = df.drop(columns=['Outcome']).values

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    }


    # a
    plt.figure(figsize=(6, 5))
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
        classification_metrics(y_test, y_pred, y_prob, label=f"{name} - a")
    plt.show()

    # b RFE
    plt.figure(figsize=(6, 5))
    for name, clf in classifiers.items():
        # Estimator for RFE
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(max_iter=1000)
        rfe = RFE(estimator, n_features_to_select=min(3, X_train_scaled.shape[1]))
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
        X_test_rfe = rfe.transform(X_test_scaled)

        clf.fit(X_train_rfe, y_train)
        y_pred = clf.predict(X_test_rfe)
        y_prob = clf.predict_proba(X_test_rfe)[:, 1] if hasattr(clf, "predict_proba") else None
        classification_metrics(y_test, y_pred, y_prob, label=f"{name} - b RFE")
    plt.show()

    # c PCA
    plt.figure(figsize=(6, 5))
    for name, clf in classifiers.items():
        pca = PCA(n_components=min(3, X_train_scaled.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        y_prob = clf.predict_proba(X_test_pca)[:, 1] if hasattr(clf, "predict_proba") else None
        classification_metrics(y_test, y_pred, y_prob, label=f"{name} - c PCA")
    plt.show()
