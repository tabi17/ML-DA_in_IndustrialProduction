from sklearn.datasets import load_digits 
from sklearn.decomposition import KernelPCA 

X, _ = load_digits(return_X_y=True) 
print('shape of initial x:', X.shape)

transformer = KernelPCA(n_components=7, kernel="rbf") 
X_transformed = transformer.fit_transform(X)

print('shape of transformed x:', X_transformed.shape)
# print(X_transformed)


