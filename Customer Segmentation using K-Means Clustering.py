# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Data
data = pd.read_csv('Mall_Customers.csv')
print(data.head())

# Preprocess the Data

# Selecting the features we want to use for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the Optimal Number of Clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
# Plotting the Elbow Curve
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means Model
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmeans.fit(X_scaled)

# Predicting the clusters
y_kmeans = kmeans.predict(X_scaled)

# Adding the cluster labels to the original dataset
data['Cluster'] = y_kmeans

# Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x= 'Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segmentation')
plt.show()

# Interpreting the Clusters

# Ensure that the 'Cluster' column is of numeric type
data['Cluster'] = data['Cluster'].astype(int)

# Grouping data by clusters to understand the characteristics
cluster_profile = data.groupby('Cluster').mean(numeric_only=True)
print(cluster_profile)
