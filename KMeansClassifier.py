import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_data(file_name):
    """
    Load dataset and return features and class labels.
    """

    dataset = pd.read_csv(file_name)

    features = dataset[["Volume", "Doors"]].values
    classes = dataset["Style"].values

    return dataset, features, classes


# --------------------------------------------------------------------------------------------------

def do_cluster(K, features, random_state):
    """
    Perform KMeans clustering using sklearn. Modified from class example.
    """

    model = KMeans(n_clusters=K, random_state=random_state)
    model.fit(features)

    labels = model.predict(features)
    centers = model.cluster_centers_

    return model, labels, centers


# --------------------------------------------------------------------------------------------------
def compute_cluster_info(K, labels, classes):
    """
    Determine majority style and accuracy per cluster.
    """

    cluster_styles = {}
    cluster_accuracy_data = []

    for cluster_number in range(K):

        cluster_data = classes[labels == cluster_number]
        size = len(cluster_data)

        if size == 0:
            majority_style = "None"
            accuracy = 0
        else:
            values, counts = np.unique(cluster_data, return_counts=True)
            majority_style = values[np.argmax(counts)]
            accuracy = np.max(counts) / size

        cluster_styles[cluster_number] = majority_style

        cluster_accuracy_data.append({
            "ClusterStyle": majority_style,
            "SizeOfCluster": size,
            "Accuracy": accuracy
        })

    return cluster_styles, cluster_accuracy_data


# --------------------------------------------------------------------------------------------------
def create_cluster_cars_csv(dataset, labels, cluster_styles):
    """
    Create ClusterCars.csv containing the original dataset with cluster style based on majority.
    """

    dataset["Cluster"] = labels
    dataset["ClusterStyle"] = dataset["Cluster"].map(cluster_styles)

    dataset[["Volume", "Doors", "Style", "ClusterStyle"]].to_csv(
        "ClusterCars.csv", index=False
    )


# --------------------------------------------------------------------------------------------------
def create_cluster_accuracy_csv(cluster_accuracy_data):
    """
    Create ClusterAccuracy.csv containing the accuracy of each cluster.
    """

    df_accuracy = pd.DataFrame(cluster_accuracy_data)
    df_accuracy.to_csv("ClusterAccuracy.csv", index=False)

# --------------------------------------------------------------------------------------------------
def main():
    K = 5
    random_state = 2

    # Load dataset (replace with your actual filename)
    dataset, features, classes = get_data("AllCars.csv")

    # Normalize data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Perform clustering
    model, labels, centers = do_cluster(K, features, random_state)

    # Compute majority styles and accuracies
    cluster_styles, cluster_accuracy_data = compute_cluster_info(K, labels, classes)

    # Create required CSV files
    create_cluster_cars_csv(dataset, labels, cluster_styles)
    create_cluster_accuracy_csv(cluster_accuracy_data)

    # Print success message
    print("ClusterCars.csv and ClusterAccuracy.csv have been created.")

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------
