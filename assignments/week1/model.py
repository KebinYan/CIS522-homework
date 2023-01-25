import numpy as np


class LinearRegression:
    """
    A linear model with a closed-form solution.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = np.zeros(1)
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # raise NotImplementedError()
        """
        Fit a closed form linear regression.
        
        Parameters:
            X (np.ndarray): The design matrix.
            y (np.ndarray): The true values.

        Returns:
            None.
        """

        weights = np.linalg.inv(X.T @ X) @ X @ y
        self.w = weights[1:]
        self.b = weights[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # raise NotImplementedError()
        """
        Makes prediction with the linear regression model.
        
        Parameters:
            X (np.ndarray): The input data.
        
        Returns:
            pred (np.ndarray): The predicted values.

        """
        pred = np.append(self.b, self.w) @ X
        return pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # raise NotImplementedError()
        """
        Fit a linear regression model using gradient descent.

        Parameters:
            X (np.ndarray): The design matrix.
            y (np.ndarray): The true value.
            lr (float): learning rate.
            epochs (int): maximum number of epochs to train the model.

        Returns:
            None.
        """
        self.w = np.zeros(X.shape[1])
        self.b = np.zeros(1)
        for epoch in range(epochs):
            y_hat = self.w @ X + self.b
            dw = -2 * np.sum(X * (y - y_hat)) / len(y)
            db = -2 * np.sum(y - y_hat) / len(y)
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()
        pred = self.w @ X + self.b
        return pred
