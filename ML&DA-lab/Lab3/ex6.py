import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv('vintage_industral_production.csv')

print("Original dataset shape:", df.shape)

# Sample becase the dataset is to big
df_sample = df.sample(n=10000, random_state=42)
print("Sampled dataset shape:", df_sample.shape)

# Select numeric + categorical columns
numeric_cols = ['OBS_VALUE']  # OBS_FLAG mostly NaN, exclude
categorical_cols = ['DATAFLOW','freq','revdate','s_adj','nace_r2','unit','geo','TIME_PERIOD']

# Fill missing
df_sample[numeric_cols] = df_sample[numeric_cols].fillna(df_sample[numeric_cols].median())
df_sample[categorical_cols] = df_sample[categorical_cols].fillna('Missing')

# Binarize numeric
df_numeric_bin = df_sample[numeric_cols].apply(lambda x: (x > x.median()).astype(int))

# One-hot encode categorical
df_categorical_bin = pd.get_dummies(df_sample[categorical_cols])

# Combine
df_bin = pd.concat([df_numeric_bin, df_categorical_bin], axis=1)

print("\nBinary dataframe sample:")
print(df_bin.head())

# Apriori
frequent_itemsets = apriori(df_bin, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence','lift'], ascending=[False, False])

print("\nApriori Association Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(
    rules['support'],
    rules['confidence'],
    s=rules['lift']*20,
    alpha=0.7
)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Apriori Association Rules – Vintages Dataset (Sample)')
plt.grid(True)
plt.show()
