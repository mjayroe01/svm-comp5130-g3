"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        wZero = np.zeros(len(self.w))
        #penalty = np.subtract(1, np.multiply(y_train, np.dot(X_train, self.w))
        if np.multiply(y_train, np.dot(X_train, self.w)) < 1: #<=?
            return np.subtract(wZero, np.multiply(self.reg_const, np.multiply(X_train, y_train)))
        else:
            return wZero

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = [0] * (len(X_train[0]))
        for i, val in enumerate(y_train):
            if val == 0:
                y_train[i] = -1
        
        for r in range(self.epochs):
            shuffler = np.random.permutation(len(y_train))
            X_train = X_train[shuffler]
            y_train = y_train[shuffler]
            for N in range(len(y_train)):
                self.w = np.subtract(self.w, np.multiply(self.alpha, self.calc_gradient(X_train[N], y_train[N])))        
        
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        pred = np.zeros(len(X_test))
        for N in range(len(X_test)):
            calc = np.dot(self.w, X_test[N])
            if calc >= 0:
                pred[N] = 1
            else:
                pred[N] = 0
        return pred
