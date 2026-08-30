import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# --- Feature Selection ---
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means Clustering ---

# From the elbow method and silhouette scores, we choose k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
df['Cluster'] = y_kmeans

# --- Analyze Clusters ---

print("Cluster Centers (Scaled):")
print(scaler.inverse_transform(kmeans.cluster_centers_))

print("\nMean values for each cluster:")
print(df.groupby('Cluster').mean(numeric_only=True))


# --- Visualize Clusters ---
plt.figure(figsize=(12, 8))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.savefig('customer_clusters.png')
plt.close()

# --- Business Insights ---

# Cluster 1 (Red): High income, low spending
# Insight: This group has money but doesn't spend much. They could be targeted with premium products or loyalty programs to encourage spending.

# Cluster 2 (Blue): Average income, average spending
# Insight: This is the largest group of customers. They are the 'average' customer. Standard marketing campaigns would work well for them.

# Cluster 3 (Green): High income, high spending
# Insight: This is the ideal customer group. They should be targeted with exclusive offers and premium products to maintain their loyalty.

# Cluster 4 (Cyan): Low income, high spending
# Insight: This group is risky. They might be overspending. They could be targeted with promotions and discounts.

# Cluster 5 (Magenta): Low income, low spending
# Insight: This group is cautious with their spending. They could be targeted with budget-friendly products and special offers.

print("\nClustering and analysis complete. Business insights are provided in the comments of the script.")