import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


df = pd.read_csv('diabetes.csv')


df_bin = df.apply(lambda x: (x > x.median()).astype(int))

# Apriori
frq_items = apriori(df_bin, min_support=0.1, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence','lift'], ascending=[False,False])

print("Apriori rules:")
print(rules)
# Vizualizare
# plt.figure(figsize=(6,4))
# plt.scatter(rules['support'], rules['confidence'], s=rules['lift']*20, alpha=0.7)
# plt.xlabel("Support")
# plt.ylabel("Confidence")
# plt.title("Apriori Rules")
# plt.grid(True)
# plt.show()
