from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris, load_diabetes
import numpy as np
import matplotlib.pyplot as plt

print("Iris Dataset:")
iris=load_iris()
class_names=iris.target_names
#X = [[0, 0], [1, 1]] 
#y = [0, 1] 
X_iris, y_iris = load_iris(return_X_y=True)

# Add non-informative features
rng = np.random.RandomState(0)
X_iris = np.hstack((X_iris, 2 * rng.random((X_iris.shape[0], 36))))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, ), random_state=1)
clf.fit(X_iris, y_iris)


# result=clf.predict([[2., 2.], [-1., -2.]])
# print('Class prediction:', result)
# result1=clf.predict_proba([[2., 2.], [1., 2.]])
# print('Class probability prediction:', result1)


this_scores = cross_val_score(clf, X_iris, y_iris)
print('Mean accuracy:', this_scores.mean())
print('Mean standard deviation:', this_scores.std())

iris=load_iris()
class_names=iris.target_names
# result=clf.predict([[5.0, 3.2, 1.3, 0.1]])
# print('Class prediction:', class_names[result[0]])

plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, edgecolor='k', label="Clase reale")
plt.title("Iris Dataset: MLPClassifier (5,2) + 36 feature-uri zgomot")
plt.legend()  #
plt.show()

#### DIABETES DATASET -->

print("Diabetes Dataset:")
diabetes=load_diabetes()
X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

rng = np.random.RandomState(0)
X_diabetes = np.hstack((X_diabetes, 2 * rng.random((X_diabetes.shape[0], 36))))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, ), random_state=1)
clf.fit(X_diabetes, y_diabetes)


# result=clf.predict([[2., 2.], [-1., -2.]])
# print('Class prediction:', result)
# result1=clf.predict_proba([[2., 2.], [1., 2.]])
# print('Class probability prediction:', result1)


this_scores = cross_val_score(clf, X_diabetes, y_diabetes)
print('Mean accuracy:', this_scores.mean())
print('Mean standard deviation:', this_scores.std())
#
# diabetes=load_diabetes()
# class_names=diabetes.target_names
# result=clf.predict([[5.0, 3.2, 1.3, 0.1]])
# print('Class prediction:', class_names[result[0]])

plt.scatter(X_diabetes[:, 0], X_diabetes[:, 1], c=y_diabetes, edgecolor='k', label="Clase reale")
plt.title("Diabetes - Datase: MLPClassifier (5,2) + 36 feature-uri zgomot")
plt.legend()  #
plt.show()


