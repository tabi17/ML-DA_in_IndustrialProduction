from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_regression 
from sklearn.metrics import  r2_score 
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X1=X[:,2]
Y1=X[:,3]
X11=X1.reshape(-1,1)
#print(X1)
#Y1.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X11, Y1, test_size=0.2, random_state=42) 
#X_train1=X_train.reshape(-1,1)
#print(X_train1)

knn_regressor = KNeighborsRegressor(n_neighbors=20)
knn_regressor.fit(X_train, Y_train) 
y_pred = knn_regressor.predict(X_test)

print("Predicted:")
print(y_pred)
r2 = r2_score(Y_test, y_pred) 
print('R-squared:', r2) 

# Visualize the results
#plt.plot([min(X_test), max(Y_test)], [min(Y_test), max(X_test)], color='red', linewidth=2, label='Ideal fit')
plt.scatter(X_test, Y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('KNN Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()