import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

class LogisticRegression:
    
    def __init__(self, learning_rate, num_iterations):
        self.alpha = learning_rate
        self.num_iterations = num_iterations
        self.weights = np.array([])
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # Rows and cols in given data set
        m = X.shape[0]
        n = X.ndim

        # Set random start weights and bias
        self.weights = np.array([random() for i in range(n+1)])

        # Train weights and bias
        for k in range(self.num_iterations):
            y_predictions = self.predict(X)

            # Update bias and weights for each data point
            for i in range(m):
                self.weights[0] += self.alpha*(y[i] - y_predictions[i])
                self.weights[1:] += self.alpha*((y[i] - y_predictions[i])*X[i])

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # Create an array filled with ones to store predictions
        y_predictions = np.array([1.0 for i in range(X.shape[0])])

        # Calculate probability-like predictions for each data points
        for i in range(X.shape[0]):
            z = sum(X[i] * self.weights[1:])
            y_predictions[i] = sigmoid(z + self.weights[0])

        return y_predictions

    def transform_to_1D(self, X):
        """
        Transforming 2D set to 1D set by calculating the

        euclidean distance between two sets of points

        Note: by passing "y=0.0", it will compute the euclidean norm

        Args:
            x, y (array<...,n>): float tensors with pairs of
                n-dimensional points

        Returns:
            A float array of shape <...> with the pairwise distances
            of each x and y point
        """
        # Transform data frame to numpy array
        X = X.to_numpy()

        # Returns the euclidean distance which will be used as the
        # new dimension
        return np.linalg.norm(X - [0, 0], ord=2, axis=-1)

# --- Some utility functions 


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

# Self made main functions for each data set
# taken most parts from Jupyter notebook

def main_data_set_1():
    # Load data set
    data_1 = pd.read_csv('data_1.csv')

    # Plot assigned data set
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', hue='y', data=data_1)

    # Partition data into independent (feature) and depended (target) variables
    X = data_1[['x0', 'x1']]
    y = data_1['y']
    plt.show()

    # Create and train model
    model_1 = LogisticRegression(0.0005, 500)
    X = X.to_numpy()
    model_1.fit(X, y)

    # Calculate accuracy and cross entropy for (insample) predictions
    y_pred = model_1.predict(X)
    print(f'Accuracy: {binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}')
    print(f'Cross Entropy: {binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}')

    # Rasterize the model's predictions over a grid
    xx0, xx1 = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
    yy = model_1.predict(np.stack([xx0, xx1], axis=-1).reshape(-1, 2)).reshape(xx0.shape)

    # Plot prediction countours along with datapoints
    _, ax = plt.subplots(figsize=(4, 4), dpi=100)
    levels = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    contours = ax.contourf(xx0, xx1, yy, levels=levels, alpha=0.4, cmap='RdBu_r', vmin=0, vmax=1)
    legends = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in contours.collections]
    labels = [f'{a :.2f} - {b :.2f}' for a,b in zip(levels, levels[1:])]
    sns.scatterplot(x='x0', y='x1', hue='y', ax=ax, data=data_1)
    ax.legend(legends, labels, bbox_to_anchor=(1,1))
    plt.show()


def main_data_set_2():
    # Load data set
    data_2 = pd.read_csv('data_2.csv')

    # Plot assigned data set
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', hue='y', data=data_2)

    # Partition data into independent (feature) and depended (target) variables
    X = data_2[['x0', 'x1']]
    y = data_2['y']
    plt.show()

    # Splitting data set into train and test
    data_2_train = data_2.query('split == "train"')
    data_2_test = data_2.query('split == "test"')

    # Partition data into independent (features) and depended (targets) variables
    X_train, y_train = data_2_train[['x0', 'x1']], data_2_train['y']
    X_test, y_test = data_2_test[['x0', 'x1']], data_2_test['y']

    # Fit model (TO TRAIN SET ONLY)
    model_2 = LogisticRegression(0.001, 500)
    X_train = model_2.transform_to_1D(X_train)
    X_test = model_2.transform_to_1D(X_test)
    model_2.fit(X_train, y_train)

    # Calculate accuracy and cross entropy for insample predictions
    y_pred_train = model_2.predict(X_train)
    print('Train')
    print(f'Accuracy: {binary_accuracy(y_true=y_train, y_pred=y_pred_train, threshold=0.5) :.3f}')
    print(f'Cross Entropy:  {binary_cross_entropy(y_true=y_train, y_pred=y_pred_train) :.3f}')

    # Calculate accuracy and cross entropy for out-of-sample predictions
    y_pred_test = model_2.predict(X_test)
    print('\nTest')
    print(f'Accuracy: {binary_accuracy(y_true=y_test, y_pred=y_pred_test, threshold=0.5) :.3f}')
    print(f'Cross Entropy:  {binary_cross_entropy(y_true=y_test, y_pred=y_pred_test) :.3f}')
