from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier 
from sklearn.datasets import load_iris 
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_iris(return_X_y=True)
rng = np.random.RandomState(0)
X = np.hstack((X, 2 * rng.random((X.shape[0], 36))))
clf1=svm.SVC(kernel='linear')
clf2 = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0).fit(X, y)

this_scores1 = cross_val_score(clf1, X, y)
print('SVM mean accuracy', this_scores1.mean())
print('SVM standard deviation of accuracies', this_scores1.std())

print("")
this_scores2 = cross_val_score(clf2, X, y)
print('Bagging SVM mean accuracy', this_scores2.mean())
print('Bagging SVM standard deviation of accuracies', this_scores2.std())