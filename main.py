# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



data = pd.read_csv('Mall-Customers/Mall_Customers.csv')  # Replace 'customer_data.csv' with your actual file path




# Select relevant columns for segmentation (e.g., Age, Annual Income, and Spending Score)
selected_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
segmentation_data = data[selected_columns]

# Convert categorical variables (like Gender) into numerical representation (e.g., 0 for Female, 1 for Male)
segmentation_data['Gender'] = segmentation_data['Gender'].map({'Female': 0, 'Male': 1})

# Scale the numerical variables (Age, Annual Income, and Spending Score) for better clustering performance
scaler = StandardScaler()
segmentation_data_scaled = scaler.fit_transform(segmentation_data)





# Perform K-means clustering with different numbers of clusters
inertia_values = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(segmentation_data_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to identify the optimal number of clusters
plt.plot(k_values, inertia_values, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()





# Choose the optimal number of clusters based on the elbow curve (e.g., 5 clusters)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(segmentation_data_scaled)

# Assign cluster labels to the original data
segmentation_data['Cluster'] = kmeans.labels_


# Examine the characteristics of each cluster
cluster_summary = segmentation_data.groupby('Cluster').mean()
print(cluster_summary)