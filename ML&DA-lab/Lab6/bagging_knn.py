
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np
from tensorflow.python.ops.gen_clustering_ops import nearest_neighbors

X, y = load_iris(return_X_y=True)
rng = np.random.RandomState(0)
X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))#zgomot

clf1=KNeighborsRegressor(n_neighbors= 5)
clf2=BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=10, random_state=0).fit(X, y)

this_scores1 = cross_val_score(clf1, X, y)
print('KNN mean accuracy', this_scores1.mean())
print('KNN standard deviation of accuracies', this_scores1.std())

print("")
this_scores2 = cross_val_score(clf2, X, y)
print('Bagging KNN mean accuracy', this_scores2.mean())
print('Bagging KNN standard deviation of accuracies', this_scores2.std())