from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd






def fugire1():
    dataset = pd.read_csv('Mall-Customers/Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    print(X)
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Clusters of Customers (Hierarchical Clustering Model)')
    plt.xlabel('Annual Income(k$)')
    plt.ylabel('Spending Score(1-100')
    plt.show()


def fugire2():
    dataset = pd.read_csv('Mall-Customers/Mall_Customers.csv')
    X = dataset.iloc[:, [2, 4]].values
    print(X)
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Clusters of Customers (Hierarchical Clustering Model)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score(1-100')
    plt.show()



def fugire3():
    dataset = pd.read_csv('Mall-Customers/Mall_Customers.csv')
    X = dataset.iloc[:, [2, 3]].values
    print(X)
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Clusters of Customers (Hierarchical Clustering Model)')
    plt.xlabel('Age')
    plt.ylabel('Annual Income(k$)')
    plt.show()



def fugire4Usingreplacemethod():
    dataset = pd.read_csv('Mall-Customers/Mall_Customers.csv')
    # replacing values
    dataset['Gender'].replace(['Male', 'Female'],
                            [0, 1], inplace=True)
    X = dataset.iloc[:, [1, 3]].values
    print(X)
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Clusters of Customers (Hierarchical Clustering Model)')
    plt.xlabel('Gender [0=>Female || 1=>Male]')
    plt.ylabel('Annual Income(k$)')
    plt.show()







def task1 ():
    fugire1()
    fugire2()
    fugire3()
    fugire4Usingreplacemethod()