from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
#from pyclustering.cluster.silhouette import 
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data  # features (4 coloane)
y = iris.target

scaler=StandardScaler()
scaled_features = scaler.fit_transform(X)


# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
amount_initial_centers = 2

amount_centers = 2
amount_candidates = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
initializer = kmeans_plusplus_initializer(scaled_features, amount_centers, amount_candidates)
centers1= initializer.initialize()
 
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
labels=encoder.get_clusters()
#labels1=labels.reshape(-1,1)
#labels1 = [[labels[j][i] for j in range (1)] for i in range (898)]

#print("Index Labeling:", labels1)

df1 = pd.DataFrame(labels)
df1.to_csv('output.csv', index=False)


xmeans_silhouette = silhouette_score(scaled_features, labels).round(2)
print('Silhouette score:', xmeans_silhouette)