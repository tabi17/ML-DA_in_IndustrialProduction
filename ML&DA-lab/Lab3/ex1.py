import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

print('\n Pima diabetes dataset -> \n')

df=pd.read_csv('diabetes.csv')
attrib_names= df.columns
# print('Attributes name: \n',attrib_names)
attrib_types=df.dtypes
# print('Attributes type: \n', attrib_types)
no_attrib=len(attrib_names)
# print('Number of attributes', no_attrib)

df = df.replace(0, np.nan) #replace 0 valued with 'nan'
# print("Intial data-frame:")
# print(df)

#from nan - > values similar
imputer = KNNImputer(n_neighbors=5)
df_imputer = imputer.fit_transform(df)   #retunrs a NumPy array not a DF
df= pd.DataFrame(df_imputer, columns=attrib_names) #convert back to pandas DF

#replace any remaining inf or NaN with column mean
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)



#ex1  PCA ->
diabetes_X=df

X_scaled = StandardScaler().fit_transform(diabetes_X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print('Original shape:', diabetes_X.shape)
print("Shape after PCA:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)


plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], color='steelblue', alpha=0.7)
plt.title('PCA - diabetes dataset')
plt.grid(True)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], color='steelblue', alpha=0.7)
ax.set_title('PCA - diabetes dataset')


####ex1  KPCA ->
from sklearn.decomposition import KernelPCA

# Apply Kernel PCA (RBF kernel)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_kpca[:,0], X_kpca[:,1], color='orange', alpha=0.7)
plt.title('KernelPCA (RBF)  - diabetes dataset')
plt.grid(True)
plt.show()


##########- varianta 2:

#### ex1 - PCA
#
# X = df
# # Standardize data
#
# X_mean = np.mean(X, axis=0)
# X_std = np.std(X, axis=0)
# X_scaled = (X - X_mean) / X_std
#
# # Covariance matrix
# cov_matrix = np.cov(X_scaled, rowvar=False)
# print("Covariance matrix shape:", cov_matrix.shape)
#
# # print(df.isna().sum())
# print('inf:\n')
# print(np.isinf(df).sum())
# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.fillna(df.mean())   # replace with column means
#
# # Eigen decomposition
# eigvals, eigvecs = np.linalg.eig(cov_matrix)
#
# # Sort by decreasing eigenvalue
# idx = np.argsort(eigvals)[::-1]
# eigvals = eigvals[idx]
# eigvecs = eigvecs[:, idx]
#
#
# # Choose number of components
# k = 2
# W = eigvecs[:, :k]
# X_pca = X_scaled @ W
#
# print("Shape after PCA:", X_pca.shape)
#
# plt.figure(figsize=(6,5))
# plt.scatter(X_pca[:,0], X_pca[:,1], color='steelblue', alpha=0.7)
# plt.title('PCA projection (from scratch)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.grid(True)
# plt.show()

print('\n Digits dataset -> \n')

from sklearn.datasets import load_digits
digits = load_digits()
digits_X = digits.data       # features (64 pixels)
digits_y = digits.target     # labels (0-9)

X_scaled = StandardScaler().fit_transform(digits_X)

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(digits_X)

print("Original shape:", diabetes_X.shape)
print("Shape after PCA:", X_pca.shape)


plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], color='steelblue', alpha=0.7)
plt.title('PCA - digits dataset')
plt.grid(True)

#KPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_kpca[:,0], X_kpca[:,1], color='orange', alpha=0.7)
plt.title('Kernel PCA (RBF) - digits dataset')
plt.grid(True)
plt.show()
