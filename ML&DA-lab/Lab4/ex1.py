import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from scipy import stats


#x - length, y width
df=pd.read_csv('iris.csv')
attrib_names= df.columns
# print(attrib_names)

x = df['petallength']
y = df['petalwidth']

#regresie liniara
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))


#polinomiala->
mymodel2 = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = np.linspace(1, 10, 100)

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.title('Regresie liniara')
plt.show()

plt.scatter(x, y)
plt.plot(myline, mymodel2(myline))
plt.title('Regresie polinomiala')
plt.show()