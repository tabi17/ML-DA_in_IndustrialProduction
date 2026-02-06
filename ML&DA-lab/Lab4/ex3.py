import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy as np

df = pandas.read_csv("diabetes.csv")
attrib_names= df.columns
print(attrib_names)

X = df[['Glucose']]
y = df['Outcome']

logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predicita pentru o singura valoare arbitrara
val_arbitrara = 120;
predicted = logr.predict(np.array([val_arbitrara]).reshape(-1,1))
print('rezultat binara daca valoarea aletoare are sau nu diabet:', predicted)

# Probabilitatea pentru toate valorile din dataset

all_probs = logr.predict_proba(X)[:, 1]
print('preedictie pentru totae vlorile din dataset:', all_probs[:10])


plt.scatter(X, y)

plt.scatter(X, all_probs)
plt.xlabel('Glucose')
plt.ylabel('Probabilitate de diabet')
plt.title('Regresie logistica - Diabetes data set')
plt.show()