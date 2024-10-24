# **Customer Segmentation using K-Means Clustering**

**Idea:** Analyze a retail dataset (e.g., customer purchase data) to identify distinct customer groups using K-means clustering. This helps in targeted marketing strategies.

# 1. Import Libraries
**You’ll need the following libraries for data manipulation, visualization, and clustering:**
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```
# 2. Load the Data
You can download a dataset like the Mall Customers Dataset or use any customer data with attributes like age, income, spending score, etc.

Here’s how you load the dataset:
```
data = pd.read_csv('Mall_Customers.csv')
print(data.head())
```
# 3. Preprocess the Data
Remove any missing values or unnecessary columns (if applicable). You can also scale the features for better clustering performance.
```
# Selecting the features we want to use for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
# 4. Finding the Optimal Number of Clusters using the Elbow Method
The Elbow Method helps in finding the number of clusters (k) by plotting the within-cluster sum of squares (WCSS) against the number of clusters.
```
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
# 5. Training the K-Means Model
Once you've identified the optimal number of clusters (e.g., 4 or 5 from the elbow method), you can fit the K-Means algorithm to the data:
```
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Predicting the clusters
y_kmeans = kmeans.predict(X_scaled)

# Adding the cluster labels to the original dataset
data['Cluster'] = y_kmeans
```
# 6. Visualizing the Clusters
You can use Seaborn or Matplotlib to visualize the clusters. Here’s an example for a 2D plot using any two features:
```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segmentation')
plt.show()
```
# 7. Interpreting the Clusters
You can analyze the resulting clusters by comparing the average characteristics (like age, income, and spending score) of each cluster:
```
# Grouping data by clusters to understand the characteristics
cluster_profile = data.groupby('Cluster').mean()
print(cluster_profile)
```

# HERE ARE THE RESULTS

![Customer Segmentation](https://github.com/user-attachments/assets/27accd83-15f6-4b25-ba40-c4ccb0745a97)
![Elbow Method](https://github.com/user-attachments/assets/c53bc5da-05ad-4861-a4df-bf0d52b1f153)
![Elbow Method 2](https://github.com/user-attachments/assets/bf93e754-4792-4ce9-82b6-bfbd28b324e9)

