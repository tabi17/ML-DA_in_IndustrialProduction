from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

#X = [[0, 0], [1, 1]] 
#y = [0, 1] 
X, y = load_iris(return_X_y=True)

# Add non-informative features
# rng = np.random.RandomState(0)
# X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))

iris=load_iris()
class_names=iris.target_names
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,), random_state=1)
clf.fit(X, y)


result=clf.predict([[5.0, 3.2, 1.3, 0.1]]) 
print('Class prediction:', class_names[result[0]])

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', label="Clase reale")
plt.title("MLPClassifier (99,) fără feature-uri suplimentare")
plt.legend()
plt.show()



