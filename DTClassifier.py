import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------------------------------
def get_car_data(file_name):
    """
    Load dataset and split into 80% train and 20% test. Modified from Class Example.
    """

    dataset = pd.read_csv(file_name)

    feature_names = ["Volume", "Doors"]
    class_name = "Style"

    features = dataset[feature_names].values
    classes = dataset[class_name].values

    class_names = sorted(dataset[class_name].unique())

    features_train, features_test, classes_train, classes_test = train_test_split(
        features,
        classes,
        test_size=0.20,
        random_state=1,
        stratify=classes
    )

    return features_train, classes_train, features_test, classes_test, feature_names, class_names

# --------------------------------------------------------------------------------------------------
def learn_tree(features_train, classes_train, criterion ,max_depth):
    """
    Train Decision Tree Classifier. From Class Example.
    """

    decision_tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=1
    )

    decision_tree.fit(features_train, classes_train)

    return decision_tree

# --------------------------------------------------------------------------------------------------
def visualise_tree(decision_tree, feature_names, class_names):
    """
    Save tree visualization as TreeCars.png. From Class Example.
    """

    plt.figure(figsize=(15,10))
    plot_tree(
        decision_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )

    plt.savefig("TreeCars.png")
    plt.close()

# --------------------------------------------------------------------------------------------------
def create_treecars_csv(features_test, classes_test, classes_predicted, accuracy):
    """
    Create TreeCars.csv including a row for accuracy.
    """

    df = pd.DataFrame(features_test, columns=["Volume", "Doors"])
    df["Style"] = classes_test
    df["PredictedStyle"] = classes_predicted

    # Add accuracy row at bottom
    accuracy_row = pd.DataFrame([{
        "Volume": "",
        "Doors": "",
        "Style": "Accuracy",
        "PredictedStyle": f"{accuracy:.4f}"
    }])

    df = pd.concat([df, accuracy_row], ignore_index=True)

    df.to_csv("TreeCars.csv", index=False)

# --------------------------------------------------------------------------------------------------
def main():
    criterion = "entropy"
    max_depth = 5

    features_train, classes_train, features_test, classes_test, feature_names, class_names = get_car_data("AllCars.csv")

    # Train tree
    decision_tree = learn_tree(features_train, classes_train, criterion, max_depth)

    # Save tree image
    visualise_tree(decision_tree, feature_names, class_names)

    # Predict on test set
    classes_predicted = decision_tree.predict(features_test)

    # Compute accuracy
    accuracy = accuracy_score(classes_test, classes_predicted)

    # Create TreeCars.csv
    create_treecars_csv(features_test, classes_test,
                        classes_predicted, accuracy)

    # Print success message
    print("TreeCars.png and TreeCars.csv have been created.")

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------------------------------