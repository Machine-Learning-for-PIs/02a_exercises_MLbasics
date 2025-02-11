from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


def compute_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy of the prediction.

    The accuracy is the most common performance measure for classification tasks.
    Is is calculated as the ration of correct predictions to all predictions.

    Args:
        y (np.ndarray): The array with the ground truth labels.
        y_pred (np.ndarray): The array with the predicted labels.
    Returns:
        float: The accuracy of the prediction.
    """
    # TODO: Implement me.
    return None


def cv_knearest_classifier(x_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    """Train and cross-validate a k-nearest neighbors classifier with the grid search.

    Define a k-nearest neighbors classifier. Use the grid search with the grid from 1 to 25
    and a 3-fold cross-validation to find the best value for the hyperparameter k.

    Args:
        x_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # create new knn model
    # TODO
    # create dictionary of all values we want to test for n_neighbors
    # TODO
    # use gridsearch and 3-fold cross-validation to test all values for n_neighbors
    # TODO
    # fit model to data
    # TODO
    return None


if __name__ == "__main__":
    # load iris dataset
    iris = load_iris()

    # print shape of data matrix and number of target entries
    # TODO
    # print names of labels and of features
    # TODO

    # create train and test split with the ratio 75:25 and print their dimensions
    # TODO

    # create a k-nn, with k = 1
    # TODO

    # train the classifier on the train set
    # TODO

    # predict labels first for train and then for test data
    # TODO

    # implement and use 'compute_accuracy' to evaluate your model by calculating accuracy on train and test set
    # TODO
    # print both accuracies
    # TODO

    # compute confusion matrix for test set
    # TODO

    # plot heatmap of confusion matrix for test set
    # TODO

    # implement and use 'cv_knearest_classifier' to perform cross validation with grid search
    # TODO

    # print top performing n_neighbors value
    # TODO

    # print mean score for the top performing value of n_neighbors
    # TODO

    # get access to best estimator
    # TODO
    # calculate train and test accuracy on best estimator
    # TODO

    # evaluate your model by calculating accuracy on train and test set
    # TODO
    # print both accuracies
    # TODO

    # plot heatmap of the confusion matrix for test set
    # TODO
