from sklearn.ensemble import AdaBoostClassifier 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
import numpy as np

X, y = load_iris(return_X_y=True)
# rng = np.random.RandomState(0)

clf = AdaBoostClassifier(n_estimators=10, random_state=0) 
#the test set will constitute 25% of the training set
X_train, X_test, y_train, y_test = train_test_split( X, y, stratify=y, random_state=42) 

accuracy=clf.fit(X_train, y_train).score(X_test, y_test) 
print(accuracy)