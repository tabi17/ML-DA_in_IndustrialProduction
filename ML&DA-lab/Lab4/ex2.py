import pandas
from sklearn import linear_model

df = pandas.read_csv("diabetes.csv")
attrib_names= df.columns
print(attrib_names)

X = df[['Glucose', 'BloodPressure']]
y = df['DiabetesPedigreeFunction']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[230, 130]])

print(predictedCO2)