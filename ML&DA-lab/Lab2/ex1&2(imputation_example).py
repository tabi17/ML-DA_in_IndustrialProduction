import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

# 1. PIMA diabetes DATASET
print('Pima diabetes dataset -> \n')
df=pd.read_csv('diabetes.csv')

attrib_names= df.columns
print('Attributes name: \n',attrib_names)
attrib_types=df.dtypes
print('\nAttributes type: \n', attrib_types)
num_attrib=len(attrib_names)
print('\nNumber of attributes', num_attrib)


df = df.replace(0, np.nan) #replace 0 valued with 'nan'


#from nan - > nearest nighbot
imputer = KNNImputer(n_neighbors=5) 
df_imputed=imputer.fit_transform(df)   #retunrs a NumPy array not a DF
print("\nFinal data-frame:")
print(df_imputed)
df= pd.DataFrame(df_imputed, columns=attrib_names) #convert back to pandas DF

print("\nCovariance Matrix - Diabetes:\n", df.cov())
print("\nCorrelation Matrix - Diabetes:\n", df.corr())



plt.matshow(df.corr(), cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Matrix - Diabetes", pad=20)


# 2.IRIS DATASET
print('iris dataset -> \n')

df_iris = pd.read_csv('iris.csv')
print('\nAtt type is \n:', df_iris.dtypes)

#Separate features and labels
X_iris = df_iris.iloc[:, :-1]  # all columns except last (features)
y_iris = df_iris.iloc[:, -1]   # last column (labels)


X_iris = X_iris.replace(0, np.nan) #replace 0 valued with 'nan'
imputer_iris = KNNImputer(n_neighbors=5)
X_iris_imputed = imputer_iris.fit_transform(X_iris)   #retunrs a NumPy array not a DF
X_iris = pd.DataFrame(X_iris_imputed, columns = X_iris.columns) #convert back to pandas DF

print("\nCovariance Matrix - Iris:\n", X_iris.cov())
print("\nCorrelation Matrix - Iris:\n", X_iris.corr())


plt.matshow(X_iris.corr(), cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Matrix - IRIS", pad=20)
plt.show()