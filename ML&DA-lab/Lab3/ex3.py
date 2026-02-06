import matplotlib.pyplot as plt
import pandas as pd
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pyclustering.cluster.xmeans import xmeans

iris = load_iris()
iris_X = iris.data  # features (4 coloane)
iris_y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris_X)

inertias = []
silhouettes = []
max_sil = 0

for i in range(1, 12):
    kmeans = KMeans(init="random", n_clusters=i, n_init=12, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    if i >= 2:
        sil = silhouette_score(X_scaled, kmeans.labels_).round(2)
        silhouettes.append(sil)
        if sil > max_sil:
            max_sil = sil

# (Elbow Method)
plt.figure(figsize=(6, 4))
plt.plot(range(1, 12), inertias, marker='o')
plt.title('K-means - Inertia (Elbow Method) on Iris')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)


plt.figure(figsize=(6, 4))
plt.plot(range(2, 12), silhouettes, marker='o')
plt.title('K-means - Silhouette on Iris')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.grid(True)

plt.show()


print('Maximum Silhouette score:', max_sil)

# Silhouette score for last fitted KMeans
kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_).round(2)
print(f'Silhouette score for {kmeans.n_clusters} clusters:', kmeans_silhouette)


#### xmeans-->


scaler=StandardScaler()
scaled_features = scaler.fit_transform(iris_X)

# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
amount_initial_centers = 2

amount_centers = 2
amount_candidates = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
initializer = kmeans_plusplus_initializer(scaled_features, amount_centers, amount_candidates)
centers1 = initializer.initialize()

# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
# number of clusters that can be allocated is 20.

xmeans_instance = xmeans(scaled_features, centers1, 20)
xmeans_instance.process()

# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()

print('length', len(centers))

# Print total sum of metric errors
print("Total WCE:", xmeans_instance.get_total_wce())

type_repr = xmeans_instance.get_cluster_encoding()
encoder = cluster_encoder(type_repr, clusters, scaled_features)

encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
print("Index Labeling:", encoder.get_clusters())
labels = encoder.get_clusters()
# labels1=labels.reshape(-1,1)
# labels1 = [[labels[j][i] for j in range (1)] for i in range (898)]

# print("Index Labeling:", labels1)

df1 = pd.DataFrame(labels)
df1.to_csv('output.csv', index=False)

xmeans_silhouette = silhouette_score(scaled_features, labels).round(2)
print('Silhouette score:', xmeans_silhouette)


#### most relevent feactures-->

from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=2)

X_new = selector.fit_transform(iris_X, iris_y)
selected_features = selector.get_support(indices=True)
print("\nRelevant Features (Iris):", [iris.feature_names[i] for i in selected_features])
