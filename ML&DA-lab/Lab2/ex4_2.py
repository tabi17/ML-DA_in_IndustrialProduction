from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("diabetes.csv")

# separare features / target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# IMPORTANT: chi2 cere valori pozitive
X = X.clip(lower=0)

# selectăm cele mai bune 2 features
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

print("X initial shape:", X.shape)
print("X after SelectKBest:", X_new.shape)

# feature names
selected_idxs = selector.get_support(indices=True)
print("\nRelevant features:")
for idx in selected_idxs:
    print("-", X.columns[idx])
