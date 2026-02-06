from sklearn import svm 
from sklearn.model_selection import cross_val_score

X = [[0, 0], [1, 1]] 
y = [0, 1]

clf = svm.SVC(kernel='rbf')
clf.fit(X, y)

result=clf.predict([[0.4, 0.4]]) #
print(result)

#this_scores = cross_val_score(clf, X, y)
#print(this_scores.mean())
#print(this_scores.std())