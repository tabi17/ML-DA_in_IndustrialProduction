from sklearn.preprocessing import StandardScaler
import pandas as pd
import kmeans_functions

customers = pd.read_csv('cl_date_financiare_tip_polita_okk1.csv')
df = pd.read_csv('cl_date_financiare_tip_polita4.csv')

scaler=StandardScaler()
scaled_features = scaler.fit_transform(customers)

inertias, silhouettes = kmeans_functions.run_kmeans(scaled_features,customers)
print("Inertia values:\n", inertias)
print("Silhouette values:\n", silhouettes)


kmeans_functions.graphics(customers, inertias, silhouettes)

#print('Inertia for 9 clusters', kmeans.inertia_)
#print(kmeans.labels_)sett

numeric_att, categorical_att = kmeans_functions.attributes_count(df)
print('numerical attribiutes are: ', numeric_att, '\ncategorical attributes are: ', categorical_att)




