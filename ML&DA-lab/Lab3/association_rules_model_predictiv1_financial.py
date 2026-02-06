# import numpy as np
# import pandas as pd
# import csv
# from mlxtend.frequent_patterns import apriori, association_rules
#
# data = pd.read_excel('cl_date_financiare_tip_polita_numeric6.xlsx')
# print(data.head())
#
# frq_items = apriori(data, min_support = 0.05, use_colnames = True)
# rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
# rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
# print('the rules are:\n',rules)
#
#

###
# Pima Diabetes example
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('diabetes.csv')

print('about the data:\n', df.head())


# Binarize numeric features
df_bin = pd.DataFrame()
df_bin['Pregnancies'] = (df['Pregnancies'] > 2).astype(int)
df_bin['Glucose'] = (df['Glucose'] > 120).astype(int)
df_bin['BloodPressure'] = (df['BloodPressure'] > 70).astype(int)
df_bin['SkinThickness'] = (df['SkinThickness'] > 20).astype(int)
df_bin['Insulin'] = (df['Insulin'] > 100).astype(int)
df_bin['BMI'] = (df['BMI'] > 30).astype(int)
df_bin['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] > 0.5).astype(int)
df_bin['Age'] = (df['Age'] > 40).astype(int)
df_bin['Outcome'] = df['Outcome']  # already 0/1

#
# df_bin = df.apply(lambda x: (x > x.median()).astype(int))


# print(df_bin.dtypes)
# print(df_bin.head())

# Apriori
frq_items = apriori(df_bin, min_support=0.1, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence','lift'], ascending=[False,False])
print('\nThe rules:\n', rules)
