import metrics
import pandas
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy as np

df = pandas.read_csv("diabetes.csv")
attrib_names= df.columns
# print(attrib_names)

X = df[['Glucose']]
y = df['Outcome']

confusion_matrix = metrics.confusion_matrix(X, y)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()
