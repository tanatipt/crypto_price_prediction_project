import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE, mutual_info_regression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# Read the CSV file containing the ETH-USD features for each day
data = pd.read_csv("../coin_data/eth_data.csv")
# We only consider data later than 2021-01-01 since this is when ETH-USD started fluctuating massively
data = data.drop(['Date'], axis=1)
# Splitting the CSV file into the feature matrix and the labels
X, y = data.drop(['Target'], axis=1), data['Target']


def create_timeseries(sequence_data, target):
    """

    A function that creates a training, validation and test set from the Cryptocurrency time series data. Each sample in the 
    dataset contains the features of the last 7 time steps and the target for the sample is the closing price of the next day.

    Args:
        sequence_data (Numpy Array): A time series data in which each row represents the features of a date
        target (Pandas Series): A Series containing the closing price of the next day

    Returns:
        Numpy Array, Numpy Array : The dataset generated by the function, The target for each saample
    """
    X = []
    y = []
    n = sequence_data.shape[0]

    # Iterating through each date with a window of length 7
    for i in range(7, n):
        # Retrieve the features of the last 7 days to create a sample
        sample_time = sequence_data[i - 7: i]
        # Retrieve the closing price of the next day from the last time step in the sample
        sample_target = target.iloc[i - 1]

        X.append(sample_time)
        y.append(sample_target)

    X = np.array(X)
    y = np.array(y)

    return X, y


def normalise(feature_importance):
    """

    This function normalises the GINI importance of each feature in the dataset when using it to 
    fit a random forest model. The feature with the highest GINI importance will be normalised to 1
    whereas the feature with the lowest GINI importance is normalised to 0

    Args:
        feature_importance (Numpy Array): An array containing the GINI importance of each feature

    Returns:
        Numpy Array: The normalised importance for each feature
    """
    f_max = np.amax(feature_importance)
    f_min = np.amin(feature_importance)

    return (feature_importance - f_min) / (f_max - f_min)


def compute_accuracy(y_test, y_pred):
    y_test_diff = np.sign(y_test)
    y_pred_diff = np.sign(y_pred)

    return (y_pred_diff == y_test_diff).sum() / len(y_test_diff)


def rf_regressor(X, y):
    """

    This function fits a random forest model with our ETH-USD time series dataset and their corresponding ground truth. The 
    random forest model is an embedded method for feature selection whereby we can extract the relevant features that contributes to 
    the model's prediction the most after fitting it. The higher the importance of the feature to the prediction, the higher the value of the 
    GINI importance. This type of feature selection method takes into account the interaction between different features in the dataset. We have 
    also performed the RFE algorithm on the Ridge and Lasso regression but the code is not shown here.

    Args:
        X (Pandas Dataframe): A Dataframe storing the ETH-USD price features for each date
        y (Pandas Series): A Series containing the next day closing price for each respective date in the dataset
    """

    # Creating a random forest regressor and fitting it with our dataset
    model = RandomForestRegressor(n_estimators=500, verbose=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    print(mean_squared_error(y, y_pred, squared=False))
    print(compute_accuracy(y, y_pred))

    plt.plot(list(range(len(y))), y)
    plt.plot(list(range(len(y_pred))), y_pred)
    plt.show()

    # Retrieving and normalising the GINI importance of each feature obtained from the fitted random forest model
    importance = normalise(model.feature_importances_)
    # Sorting the features based on the GINI importance score
    best_idx = np.argsort(importance)[::-1]
    best_scores = np.sort(importance)[::-1]

    names = data.columns.values[0:-1]
    ticks = [i for i in range(len(names))]

    print(names[best_idx])
    print(best_scores)

    # Plotting the GINI importance of each feature using a bar graph and then displaying it
    plt.bar(names, importance)

    for i in range(len(names)):
        plt.text(i, importance[i], round(importance[i], 2), ha='center')
    plt.xticks(ticks, names, rotation=90)
    plt.show()


def rfe(X, y):
    """

    This function performs recursive feature elimination to rank the best features in the dataset one-by-one. RFE is a wrapper method
    for feature selection whereby it considers the selection of the optimal feature sets as a search problem. In our scenario, we are 
    recursively fitting a random forest model with our dataset and removing the feature with lowest importance in each iteration. This process
    is repeated until we end up with a single feature. At the end , these features are ranked according to when they are pruned by the algorithm.

    Args:
        X (Pandas Dataframe): A Dataframe storing the ETH-USD price features for each date
        y (Pandas Series): A Series containing the next day closing price for each respective date in the dataset
    """

    # Applying the RFE algorithm on the random forest regressor. The RFE uses the GINI importance of the random forest to eliminate the weakest feature.
    rfe = RFE(RandomForestRegressor(n_estimators=200), n_features_to_select=1)
    fit = rfe.fit(X, y)

    names = data.columns.values[0:-1]
    ticks = [i for i in range(len(names))]

    # Retrieving the ranking of each feature obtained from the fitted random forest model with the RFE algorithm
    rankings = fit.ranking_
    # Plotting the ranking of each feature using a bar graph and then displaying it
    plt.bar(ticks, fit.ranking_)

    for i in range(len(names)):
        plt.text(i, rankings[i], rankings[i], ha='center')

    plt.xticks(ticks, names, rotation=90)
    plt.show()


def spearman_coefficient(X, y):
    """

    This function computes the spearman coefficient between each feature and the target variable. Spearman coefficient is a 
    filter approach for feature selection where the features are selected using statistical methods This method is independent of
    the learning algorithm so it is computationally inexpensive but does not consider the interaction between the features. A coefficient of 1 indicates 
    a perfect positive association between the feature and the target variable whereas a coefficient of -1 indicates a perfect negative association.
    The coefficent is always between -1 and +1.

    Args:
        X (Pandas Dataframe): A Dataframe storing the ETH-USD price features for each date
        y (Pandas Series): A Series containing the next day closing price for each respective date in the dataset
    """

    rhos = []
    ps = []
    names = data.columns.values[0:-1]
    ticks = [i for i in range(len(names))]

    # Iterating to compute the coefficent between each feature and the target variable
    for i in range(X.shape[1]):
        rho, p = spearmanr(X[:, i], y)
        # Recording the coefficent and p-value of the feature with the target variable
        rhos.append(rho)
        ps.append(p)

    # Plotting the spearman coefficent of each feature into a bar graph wrt the target variable
    plt.bar(ticks, rhos)

    for i in range(len(names)):
        plt.text(i, rhos[i], round(rhos[i], 3), ha='center')

    plt.xticks(ticks, names, rotation=90)
    plt.show()

    # Plittign the p-value of each feature into a bar graph wrt the target variable
    plt.bar(ticks, ps)
    for i in range(len(names)):
        plt.text(i, ps[i], round(ps[i], 3), ha='center')

    plt.xticks(ticks, names, rotation=90)
    plt.show()

    rhos = np.array(rhos)
    best_idx = np.argsort(np.abs(rhos))[::-1]
    print(names[best_idx])
    print(rhos[best_idx])


def mutual_info(X, y):
    """

    Mutual information is another filter method for feature selection. Mutual regresson measures the dependency between the target variable
    and a feature in the dataset. The higher the value, the higher the dependency between the feature and the target; hence, it is more important
    in predicting the target variable.


    Args:
        X (Pandas Dataframe): A Dataframe storing the ETH-USD price features for each date
        y (Pandas Series): A Series containing the next day closing price for each respective date in the dataset
    """
    names = data.columns.values[0:-1]

    # Computing the mutual information value between each feature and the target value
    scores = mutual_info_regression(X, y)

    # Soring the features according to the mutual information value
    best_idx = np.argsort(scores)[::-1]
    best_score = np.sort(scores)[::-1]
    ticks = [i for i in range(len(names))]

    print(names[best_idx])
    print(best_score)

    # Plotting the features in descending order of mutual information value
    plt.bar(ticks, best_score)
    plt.xticks(ticks, names[best_idx], rotation=90)
    plt.show()


# 'Close' 'VWAP' 'Low' 'High' 'EMAF' 'VISIBLE_ITS' 'BBL' 'OBV'
rf_regressor(X, y)
# Close, VWAP, Low, High, VISIBLE_ITS, OBV, BBL, MACD_SIGNAL, EMAF
rfe(X, y)
# Close, VWAP, High, Low, Open, EMAF, BBM , VISIBLE_ITS, BBL
spearman_coefficient(X, y)
# Close, VWAP, High, Low, Open, EMAF, BBM , BBL, VISIBLE_ITS, BBU, VISIBLE_IKS, EMAM
mutual_info(X, y)

# Final Features : Close, VWAP, Low, High, EMAF, VISIBLE_ITS, BBL, BBM, OBV, BBM
