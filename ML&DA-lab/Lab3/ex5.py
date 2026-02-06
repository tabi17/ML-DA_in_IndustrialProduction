# K-MODES & K-PROTOTYPES CLUSTERING
# Vintages Industrial Production dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

customers = pd.read_csv("vintage_industral_production.csv")

print("Original dataset shape:", customers.shape)


# Random sampling (10,000 instances) to reduce dataset size
customers_sample = customers.sample(n=10000, random_state=42)

print("Sampled dataset shape:", customers_sample.shape)

# Numeric columns
numeric_cols = customers_sample.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

# Categorical columns
categorical_cols = customers_sample.select_dtypes(
    include=['object', 'category', 'bool']
).columns.tolist()

print("\nInitial numeric columns:", numeric_cols)
print("Initial categorical columns:", categorical_cols)

#preproces
numeric_cols = [
    col for col in numeric_cols
    if customers_sample[col].notna().sum() > 0
]

print("\nFiltered numeric columns:", numeric_cols)


# Numeric -> fill NaN with median
for col in numeric_cols:
    median_value = customers_sample[col].median()
    customers_sample[col] = customers_sample[col].fillna(median_value)

# Categorical -> fill NaN with explicit category
for col in categorical_cols:
    customers_sample[col] = customers_sample[col].fillna('Missing')



# K-MODES CLUSTERING (categorical data only)

# Convert categorical columns to string (required)
X_cat = customers_sample[categorical_cols].astype(str)

kmodes = KModes(
    n_clusters=3,
    init='Huang',
    n_init=3,
    random_state=42
)

customers_sample['cluster_kmodes'] = kmodes.fit_predict(X_cat)

print("\nK-Modes cluster distribution:")
print(customers_sample['cluster_kmodes'].value_counts())



# K-PROTOTYPES CLUSTERING (mixed data)

# Combine numeric + categorical columns
X_mix = customers_sample[numeric_cols + categorical_cols]
X_mix_np = X_mix.to_numpy()

# Indices of categorical columns in mixed array
categorical_idx = [
    i for i, col in enumerate(numeric_cols + categorical_cols)
    if col in categorical_cols
]

kproto = KPrototypes(
    n_clusters=3,
    init='Huang',
    n_init=3,
    random_state=42
)

customers_sample['cluster_kprototypes'] = kproto.fit_predict(
    X_mix_np,
    categorical=categorical_idx
)

print("\nK-Prototypes cluster distribution:")
print(customers_sample['cluster_kprototypes'].value_counts())

# plt.figure(figsize=(8, 5))
# plt.scatter(
#     customers_sample['OBS_VALUE'],
#     customers_sample['cluster_kprototypes'],
#     c=customers_sample['cluster_kprototypes'],
#     alpha=0.5
# )
#
# plt.xlabel("OBS_VALUE")
# plt.ylabel("Cluster (K-Prototypes)")
# plt.title("K-Prototypes Clustering")
#
# plt.show()
#




# print(customers_sample.head())


# customers_sample.to_csv(
#     "vintage_industral_production_clustered_sample.csv",
#     index=False
# )
#
# print("\nClustered dataset saved successfully.")
