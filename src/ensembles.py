import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import time

class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None, return_train_loss=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.algorithms = []
        times = []
        if return_train_loss is True:
            train_loss = []
        if X_val is not None and y_val is not None:
            val_loss = []
        for i in range(self.n_estimators):
            idx = np.random.randint(X.shape[0], size=X.shape[0])
            model = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters) 
            start = time.time()
            model.fit(X[idx], y[idx])
            end = time.time() - start
            times.append(end)
            self.algorithms.append(model)
            if return_train_loss is True:
                train_pred = np.mean([m.predict(X) for m in self.algorithms], axis=0)
                train_loss.append(np.sqrt(((y - train_pred) ** 2).mean()))
            if X_val is not None and y_val is not None:
                val_pred = np.mean([m.predict(X_val) for m in self.algorithms], axis=0)
                val_loss.append(np.sqrt(((y_val - val_pred) ** 2).mean()))
        if X_val is not None and y_val is not None and return_train_loss is True:
            return train_loss, val_loss, np.cumsum(times)
        elif X_val is not None and y_val is not None:
            return val_loss, np.cumsum(times)
        elif return_train_loss is True:
            return train_loss, np.cumsum(times)


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = [model.predict(X) for model in self.algorithms]
        return np.mean(ans, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def MSE(coef, y, pred, new_pred):
        return ((y - pred - coef * new_pred) ** 2).mean()


    def fit(self, X, y, X_val=None, y_val=None, return_train_loss=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        self.algorithms = []
        self.coef = []
        times = []
        pred = np.zeros(X.shape[0])
        if return_train_loss is True:
            train_loss = []
        if X_val is not None and y_val is not None:
            val_loss = []
            val_pred = np.zeros(X_val.shape[0])

        def MSE(coef, y, pred, new_pred):
            return ((y - pred - coef * new_pred) ** 2).mean()

        for i in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            start = time.time()
            model.fit(X, 2 * (pred - y) / y.shape[0])
            end = time.time() - start
            times.append(end)
            new_pred = model.predict(X)
            self.coef.append(minimize_scalar(MSE, args=(y, pred, new_pred)).x)
            pred += self.learning_rate * self.coef[-1] * new_pred
            self.algorithms.append(model)
            if return_train_loss is True:
                train_loss.append(np.sqrt(((y - pred) ** 2).mean()))
            if X_val is not None and y_val is not None:
                new_val_pred = model.predict(X_val)
                val_pred += self.learning_rate * self.coef[-1] * new_val_pred
                val_loss.append(np.sqrt(((y_val - val_pred) ** 2).mean()))
        if X_val is not None and y_val is not None and return_train_loss is True:
            return train_loss, val_loss, np.cumsum(times)
        elif X_val is not None and y_val is not None:
            return val_loss, np.cumsum(times)
        elif return_train_loss is True:
            return train_loss, np.cumsum(times)


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            ans += self.learning_rate * self.coef[i] * self.algorithms[i].predict(X)
        return ans
