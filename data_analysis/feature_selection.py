from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ml_system import rescale_data
from itertools import combinations
from sklearn.base import clone
from sklearn.svm import SVR
import numpy as np
import sqlite3
import pandas as pd

original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('../db_work/ApartmentsInfo.db'))\
    # .drop(columns=['index'])

original_dataset["ceiling_height"] = pd.to_numeric(original_dataset['ceiling_height'])

original_dataset = original_dataset.dropna()


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


def select_features_for_rooms_predictions():
    data = rescale_data(original_dataset.drop(columns='rooms'))
    features_names = data.drop(columns='rooms').columns
    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=5, scoring=accuracy_score)

    sbs.fit(data.values, original_dataset['rooms'].values)

    most_important_features = [features_names[i] for i in sbs.result[max(sbs.result, key=sbs.result.get)]]

    return most_important_features


def select_features_for_cost_predictions():
    data = rescale_data(original_dataset)
    features_names = data.drop(columns='cost').columns

    estimator = SVR(kernel='rbf', C=1e3, gamma=0.1)
    sbs = SBS(estimator, k_features=5)
    sbs.fit(data.drop(columns='cost').values, data['cost'].values)

    most_important_features = [features_names[i] for i in sbs.result[max(sbs.result, key=sbs.result.get)]]

    return most_important_features


select_features_for_cost_predictions()