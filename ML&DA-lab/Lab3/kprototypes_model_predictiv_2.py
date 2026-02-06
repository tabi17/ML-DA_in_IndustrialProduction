#Aplicarea metodei k-prototypes clustering in Python, pe seturi de date mixte:   
#numerice si categoriale
import numpy as np 
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes

customers = pd.read_csv('vintage_industral_production.csv')
costs=[]
n_clusters=[]
clusters_assigned=[]

for i in range(3, 10):
        kproto = KPrototypes(n_clusters=i, init='Huang', random_state=0)
        clusters = kproto.fit_predict(customers, categorical=[3])
        costs.append(kproto.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)

layout = go.Layout(
   #title = 'Cost/Nr. clusteri',
   xaxis = dict(
      title = 'Nr. clusteri',zeroline=True,
      showline = True
   ),
   yaxis = dict(
      title = 'Cost',zeroline=True,
      showline = True
   ))
fig = go.Figure(data=go.Scatter(x=n_clusters, y=costs, name='Cost/Nr. clusteri'), layout=layout)
fig.show()
print(pd.Series(clusters).value_counts())
plt.figure(figsize=(12,5))
np.random.seed(19680801)
area = 30
customers['label'] = kproto.labels_
print(customers['label'])
colors=customers['label']

plt.figure(1)
plt.scatter(x=customers['judet'],y=customers['tip_polita'], s=area, c=colors, alpha=0.5)
plt.xlabel("judet")
plt.ylabel("tip_polita")

plt.figure(2)
plt.scatter(x=customers['is_pensionar'],y=customers['tip_polita'], s=area, c=colors, alpha=0.5)
plt.xlabel("pensionar(0=nu/1=da)")
plt.ylabel("tip_polita")
plt.show()