from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=self.test_size , random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self.calc_score(X_train, y_train , X_test , y_test , self.indices_)
        self.scores = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self . indices_, r=dim-1):
                score = self.calc_score(X_train, y_train, X_test, y_test , р)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores.append(scores[best])
        self.k_score = self.scores[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score