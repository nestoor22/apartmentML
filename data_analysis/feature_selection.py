from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from itertools import combinations
from .scalers import rescale_data
from sklearn.base import clone
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import sqlite3


class SBS:
    def __init__(self, estimator, k_features, scoring=None, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.result = {}

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self.calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = int(np.argmax(scores))
            self.indices_ = subsets[best]
            self.result[scores[best]] = subsets[best]
            print(self.result)
            dim -= 1

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])

        if self.scoring:
            score = self.scoring(y_test, y_pred)
        else:
            score = self.estimator.score(X_test[:, indices], y_pred)

        return score


def select_features_for_class_predictions(data, what_predict):
    features_names = data.drop(columns=what_predict).columns

    new_data, _ = rescale_data(data.drop(columns=what_predict))
    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=5, scoring=accuracy_score)

    sbs.fit(new_data.values, data[what_predict].values)

    most_important_features = [features_names[i] for i in sbs.result[max(sbs.result, key=sbs.result.get)]]

    return most_important_features


def select_features_for_regression_predictions(data, what_predict):
    data, _ = rescale_data(data)

    features_names = data.drop(columns=what_predict).columns

    estimator = SVR(C=1e3, gamma=0.01)
    sbs = SBS(estimator, k_features=5)
    sbs.fit(data.drop(columns=what_predict).values, data[what_predict].values)

    most_important_features = [features_names[i] for i in sbs.result[max(sbs.result, key=sbs.result.get)]]

    return most_important_features


def select_features_for_class_prediction_with_trees(data, what_predict):
    data, _ = rescale_data(data)

    tree = ExtraTreesClassifier(n_estimators=15)
    tree.fit(data.values, original_dataset[what_predict].values)

    feature_importance = list(tree.feature_importances_)
    most_important_features = []
    for i in range(5):
        arg_max = np.argmax(feature_importance)
        most_important_features.append(list(data.columns).pop(arg_max))
        feature_importance[arg_max] = 0

    return most_important_features


def select_features_for_reg_prediction_with_trees(data, what_predict):
    data, _ = rescale_data(data)

    tree = DecisionTreeRegressor()
    tree.fit(data.drop(columns=what_predict).values, data[what_predict].values)
    data = data.drop(columns=what_predict)

    feature_importance = list(tree.feature_importances_)
    most_important_features = []
    for i in range(5):
        arg_max = np.argmax(feature_importance)
        most_important_features.append(list(data.columns).pop(arg_max))
        feature_importance[arg_max] = 0

    return most_important_features


def remove_highly_correlated_features(dataset):
    correlation_matrix = dataset.corr()

    # upper triangle of matrix
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    columns_to_remove = [column for column in upper if any(upper[column].abs() > 0.8)]
    return dataset.drop(columns=columns_to_remove)
