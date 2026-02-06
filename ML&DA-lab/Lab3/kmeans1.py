import matplotlib.pyplot as plt
#from kneed import KneeLocator
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

df = pd.read_csv('diabetes.csv')

scaler=StandardScaler()
scaled_features = scaler.fit_transform(df)

inertias=[]
silhouette=[]
max_sil=0

for i in range(1,12):
    kmeans = KMeans(init="random", n_clusters=i, n_init=12, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)
    if i>=2:
     sil=silhouette_score(scaled_features, kmeans.labels_).round(2)
     silhouette.append(sil)
     if sil>max_sil:
       max_sil=sil


plt.figure(1)
plt.plot(range(1,12), inertias, marker='o')
plt.title('K-means - Inertias')
plt.xlabel('Nr clusteri')
plt.ylabel('Inertia')
plt.grid(True)

plt.figure(2)
plt.plot(range(2,12), silhouette, marker='o')
plt.title('K-means - Silhouette')
plt.xlabel('Nr clusteri')
plt.ylabel('Silhouette score')
plt.grid(True)

plt.show()

print('Maximum Silhouette score:', max_sil)

kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(2)
print('Silhouette score for 15 clusters:', kmeans_silhouette)
