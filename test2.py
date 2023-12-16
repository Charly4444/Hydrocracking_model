import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('fulldata.csv')

# Summary statistics and data overview
# print(data.describe())
# print(data.min(axis='rows'))

data['Naphtha'] = abs(data['Naphtha'])
data['Diesel'] = abs(data['Diesel'])
data['Kero'] = abs(data['Kero'])

data['Naphtha(%)'] = (data.apply(lambda row: row['Naphtha']/0.077, axis=1))*100
data['Diesel(%)'] = (data.apply(lambda row: row['Diesel']/0.680, axis=1))*100
data['Kero(%)'] = (data.apply(lambda row: row['Kero']/0.113, axis=1))*100

# save modified file
data.to_csv('fulldata_m.csv')

# ==============================================================================

data = pd.read_csv('fulldata_m.csv')

#print(data.head())
#velocity	T(K)	Naphtha	Diesel	Kero
selected_columns = [
                    'velocity',
                    'T(K)',
                    'Naphtha',
                    'Diesel',
                    'Kero'
                    ]

selected_data = data[selected_columns]

# Pairplot for visualizing relationships
sns.pairplot(selected_data)
# plt.show()
# save up
plt.savefig("gen_IMAGES/pairplt.png")

# Correlation matrix and heatmap
correlation_matrix = selected_data.corr()
plt.figure(figsize=(10, 8))
heatplt = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
# plt.show()
# save up
heatplt.get_figure().savefig('gen_IMAGES/heatplt.png')

# Distribution of individual variables
plt.figure(figsize=(12, 8))
for col in selected_data.columns:
    sns.histplot(selected_data[col], kde=True, label=col)
plt.legend()
plt.title("Distribution of Variables")
plt.show()
