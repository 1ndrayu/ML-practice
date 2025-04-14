import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

d = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv")
X = d[['Annual Income (k$)', 'Spending Score (1-100)']]
m = KMeans(n_clusters=5, random_state=42)
d['c'] = m.fit_predict(X)

plt.figure(figsize=(8,6))
for i in range(5):
    g = d[d['c'] == i]
    plt.scatter(g['Annual Income (k$)'], g['Spending Score (1-100)'], label=f'C{i}')
plt.xlabel("Income")
plt.ylabel("Score")
plt.title("Clusters")
plt.legend()
plt.grid()
plt.show()
