from sklearn import svm 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris, load_diabetes
import numpy as np
import matplotlib.pyplot as plt

print("Iris dataset:\n")

iris = load_iris()
#X = [[0, 0], [1, 1]] 
#y = [0, 1] 
X, y = load_iris(return_X_y=True)

# Add non-informative features
rng = np.random.RandomState(0)
X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))# in total 40

clf = svm.SVC(kernel='linear')#
clf.fit(X, y)

#result=clf.predict([[2., 2.]])
#print(result)
this_scores = cross_val_score(clf, X, y)
print('Mean accuracy:', this_scores.mean())
print('Mean standard deviation:', this_scores.std())


######

print('\nDiabetes dataset:\n')

diabetes=load_diabetes()
X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

rng = np.random.RandomState(0)
X_diabetes = np.hstack((X_diabetes, 2 * rng.random((X_diabetes.shape[0], 36))))# in total 40

clf = svm.SVC(kernel='linear')#
clf.fit(X_diabetes, y_diabetes)

#result=clf.predict([[2., 2.]])
#print(result)
this_scores = cross_val_score(clf, X_diabetes, y_diabetes)
print('Mean accuracy:', this_scores.mean())
print('Mean standard deviation:', this_scores.std())



