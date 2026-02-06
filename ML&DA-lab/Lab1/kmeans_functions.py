import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(scaled_features, customers):
 """
 Runs K-means for k = 1..9, computes inertia & silhouette scores,
 and assigns the best label to the customers dataframe.
 """
 max_sil=0
 inertias = []
 silhouettes = []
 best_silhouette = -1
 best_labels = None

 for i in range(1,10): #11
    model = KMeans(init="random", n_clusters=i, n_init=10, max_iter=300, random_state=42)
    model.fit(scaled_features)
    inertias.append(model.inertia_)


    if i>=2:
     #sil =silhouette_score(scaled_features, model.labels_).round(2) #metoda numpy (rotunjeste la 2 zecimale)
     sil = round(silhouette_score(scaled_features, model.labels_), 2) # fc python
     silhouettes.append(sil)

     if sil>best_silhouette:
        best_silhouette = sil
        best_labels = model.labels_

 print(f"Max silhouette score: {best_silhouette}")
 customers['label'] = best_labels

 return inertias, silhouettes



def graphics(customers, inertias, silhouette):
    # --- Inertia plot ---
 plt.figure(1)
 plt.plot(range(1,10), inertias, marker='o')
 plt.title('K-means - Inertias')
 plt.xlabel('Numb of clusters')
 plt.ylabel('Inertia')
 plt.grid(True)
    # --- Silhouette plot ---
 plt.figure(2)
 plt.plot(range(2,10), silhouette, marker='o')
 plt.title('K-means - Silhouette')
 plt.xlabel('Numb of clusters')
 plt.ylabel('Silhouette score')
 plt.grid(True)
 #plt.show()
    # --- Cluster scatter plot ---
 colors=customers['label']
 area = 30
 plt.figure(3)
 plt.scatter(x=customers['is_pensionar'],y=customers['tip_polita'], s=area, c=colors, alpha=0.5)
 plt.xlabel("is_retired")
 plt.ylabel("tip_polita")
 plt.grid(True)
 plt.show()

def attributes_count(df):
 """
 Counts numerical vs categorical attributes.
 """
 numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
 categorical_cols = df.select_dtypes(include=['object', 'category']).columns

 num_numeric = len(numeric_cols)
 num_categorical = len(categorical_cols)

 return num_numeric, num_categorical


