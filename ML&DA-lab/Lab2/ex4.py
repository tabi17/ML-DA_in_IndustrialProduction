from sklearn.datasets import load_iris #load_digits 
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Iris dataset
print('Iris dataset -> \n')
X_iris, y_iris = load_iris(return_X_y=True) #load_digits
iris=load_iris()
class_names=iris.target_names
attribute_names=iris.feature_names

selector=SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X_iris, y_iris) #20

print("X initial shape", X_iris.shape)
print("Shape after SelectKBest:", X_new.shape)


print("class names", class_names)
print("class codes", y_iris)
print("feature names", attribute_names)



cols_idxs = selector.get_support(indices=True)
print("\nRelevant features:")

for idx in cols_idxs:
    print("-", attribute_names[idx])


#
# Pima diabetes dataset
print('\nPima diabetes dataset -> \n')
df = pd.read_csv("diabetes.csv")

X_diabetes = df.drop("Outcome", axis=1)
y_diabetes = df["Outcome"]

# chi2 cere valori pozitive
X_diabetes = X_diabetes.clip(lower=0)

selector = SelectKBest(score_func=chi2, k=2)
Xnew = selector.fit_transform(X_diabetes, y_diabetes)

print("X initial shape:", X_diabetes.shape)
print("X after SelectKBest:", Xnew.shape)

selected_idxs = selector.get_support(indices=True)
print("\nRelevant features:")
for idx in selected_idxs:
    print("-", X_diabetes.columns[idx])