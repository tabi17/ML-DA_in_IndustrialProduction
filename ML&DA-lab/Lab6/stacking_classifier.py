from sklearn.datasets import load_iris
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)

# Define base estimators
estimators = [
    ('svc', make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Define stacking classifier with Logistic Regression as meta-learner
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method='auto'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# Train and evaluate
accuracy = clf.fit(X_train, y_train).score(X_test, y_test)
print("Accuracy:", accuracy)
