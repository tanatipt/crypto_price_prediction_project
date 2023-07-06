import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor


def compute_accuracy(y_test, y_pred):
    y_test_sign = np.sign(y_test)
    y_pred_sign = np.sign(y_pred)

    return (y_pred_sign == y_test_sign).sum() / len(y_test_sign)


def chronological_split(X, y, percentage):
    size = int(percentage * X.shape[0])
    X_1, y_1 = X.iloc[:size], y.iloc[:size]
    X_2, y_2 = X.iloc[size:], y.iloc[size:]

    return X_1, X_2, y_1, y_2


def rf_bayes_search(X_aux, y_aux):
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier()

    param_grid = {"n_estimators": [
        100, 200, 300, 400], 'max_depth': [4, 8, 12, 16], 'criterion': ["gini", "entropy", "log_loss"]}
    clf = HalvingGridSearchCV(
        model, param_grid=param_grid,  cv=tscv, return_train_score=True, verbose=3, error_score="raise", scoring="accuracy")
    clf.fit(X_aux, y_aux)

    print(clf.best_params_)
    print(clf.best_score_)


eth_data = pd.read_csv("../../coin_data/eth_data.csv")
eth_data = eth_data.drop(['Date'], axis=1)
sequence_data, target = eth_data.drop(['Target'], axis=1), eth_data['Target']
X_aux, X_test, y_aux, y_test = chronological_split(sequence_data, target, 0.75)

print(X_aux)
print(y_aux)
rf_bayes_search(X_aux, y_aux)
