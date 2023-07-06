import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import keras_tuner as kt
from MyHyperModel import MyHyperModel

# 2 LSTM Cells, 2 Dense Layers
# {'lstm_1': 32, 'lstm_2': 64, 'dense_1': 128, 'dense_2': 8, 'dropout': 0.0, 'learning_rate': 0.001,
# 'weight_initialiser': 'he_normal', 'activation': 'leaky_relu', 'epochs': 250, 'batch': 8} 62.911781311035156

# 1 LSTM Cell, 2 Dense Layers
# {'lstm_1': 32, 'dense_1': 8, 'dense_2': 16, 'dropout': 0.0, 'learning_rate': 0.001, 'weight_initialiser':
# 'he_normal', 'activation': 'relu', 'epochs': 250, 'batch': 8} 62.92028045654297

# 2 LSTM Cell , 1 Dense Layers
# { 'lstm_1': 8, 'lstm_2': 8, 'dense_1': 32, 'dropout': 0.0, 'learning_rate': 0.01,
# 'weight_initialiser': 'random_normal', 'activation': 'leaky_relu', 'epochs': 200, 'batch': 8} 63.54599380493164

# 1 LSTM Cell , 1 Dense Layers
# {'lstm_1': 32, 'dense_1': 128, 'dropout': 0.0, 'learning_rate': 0.001,
# 'weight_initialiser': 'he_normal', 'activation': 'elu', 'epochs': 150, 'batch': 8} 63.23390579223633


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


def chronological_split(X, y, percentage):
    """

    Chronologically split X and y into 2 sets where all the samples in the first one precede all samples
    in the second one. Given a percentage x, the first set contains x% of the original samples 
    whereas the second set contains (1-x)% of the remaining samples

    Args:
        X (Pandas Dataframe): A time series data in which each row represents the features of a date
        y (Pandas Series): A Series containing the closing price of the next day
        percentage (Float): The percentage of samples to retain in the first set

    Returns:
        Pandas Dataframe, Pandas Dataframe, Pandas Series, Pandas Series : The first set samples, the first set targets, the second set samples, the second set targets
    """
    size = int(percentage * X.shape[0])
    X_1, y_1 = X.iloc[:size], y.iloc[:size]
    X_2, y_2 = X.iloc[size:], y.iloc[size:]

    return X_1, X_2, y_1, y_2


def min_max_scale(X_1, X_2):
    """

    Performs a MinMax scaling on 2 datasets. The scaler is first fitted using the dataset X_1 and then X_1 is transformed. After this, 
    the fitted scaler is used to transform X_2. X_1 is usually the training set and X_2 is usually the validation/test set

    Args:
        X_1 (Pandas Dataframe): The training set
        X_2 (Pandas Dataframe): The validation/test set

    Returns:
        Numpy Array, Numpy Array: The transformed training set, the transformed validation/test set.
    """
    scaler = MinMaxScaler()
    X_1 = scaler.fit_transform(X_1)
    X_2 = scaler.transform(X_2)

    return X_1, X_2


def standard_scale(y_1, y_2):
    scaler = StandardScaler()
    y_1 = scaler.fit_transform(y_1.values.reshape(-1, 1)).flatten()
    y_2 = scaler.transform(y_2.values.reshape(-1, 1)).flatten()

    return y_1, y_2


def lstm_bayes_search(X_aux, y_aux):
    """

    Performs Bayesian search on the LSTM neural network model to find the best hyperparameter combination using Keras Tuner and the Hypermodel
    .Initially, the dataset X_aux, y_aux is splitted into the training set , X_train and y_train and the validation set , X_test, y_test. After this, 
    both the validation and training dataset are normalised using the min_max_scale function. We are using a hold-out validation set to evaluate each
    hyperparameter combination after training

    Args:
        X_aux (Pandas Dataframe): A dataset that will used to create the training and validation set
        y_aux (Pandas Series): A Series containing the closing price of the next day for each date in X_aux

    Returns:
        Keras Model, Keras Hypermodel, Keras Hyperparameter : Keras model fitted with the optimal hyperparameters, Keras hypermodel that was evaluated, The optimal hyperparameter evaluated
    """

    # Splitting the initial dataset into the training set and validation set with a relative proportion of 2/3 and 1/3
    X_train, X_val, y_train, y_val = chronological_split(X_aux, y_aux, 2/3)
    # Scaling both the training and validation dataset
    X_train, X_val = min_max_scale(X_train, X_val)
    # y_train, y_val = standard_scale(y_train, y_val)
    # Creating a 3-dimensional dataset for the training and validation set where each sample consists of the 7 previous time steps
    X_train, y_train = create_timeseries(X_train, y_train)
    X_val, y_val = create_timeseries(X_val, y_val)

    # Defining a Keras Tuner object using Bayesian optmisation. max_trials indicate the number of hyperparmaeter combination to be evaluated
    # and the objective to be minimised during the search is the validation loss.
    tuner = kt.BayesianOptimization(
        MyHyperModel(),
        objective="val_loss",
        max_trials=5,
        overwrite=True,
        project_name="test"
    )

    # Performing the Bayesian search across the hyperparameters distribution on the training set. After training a model on a specific hyperparameter combination,
    # the trained model is evaluated on the validation set. An early stopping method has been used to prevent overfitting and divergence from the minimum validation loss.
    tuner.search(X_train, y_train, validation_data=(X_val, y_val),
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")])

    # Obtaining hyperparameter combination that produced the lowest validation loss
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)
    # Reconstructing and fitting the LSTM neural network with the optimal hyperparameters
    hypermodel = MyHyperModel()
    model = hypermodel.build(best_hp)
    return model, hypermodel, best_hp


def compute_accuracy(y_test, y_pred):
    y_pred = y_pred.flatten()
    y_test_diff = np.sign(y_test).flatten()
    y_pred_diff = np.sign(y_pred)

    return (y_pred_diff == y_test_diff).sum() / len(y_test_diff)


def train_test(model, hypermodel, best_hp, X_aux, X_test, y_aux, y_test):
    """

    After hyperparamter selection, we fit the model with the best hyperparameters on the entire
    initial dataset X_aux and then evaluate its performance on the test set X_test. Both X_aux and X_test
    are normalised and transformed first before the fitting and evaluation process.

    Args:
        model (Keras Model): The LSTM neural network model fitted with the optimal hyperparamters
        hypermodel (Keras Hypermodel): The Hypermodel that evaluated the hyperparameters for the LSTM network
        best_hp (Keras Hyperparameters): A Hyperparameters object that stores the optimal hyperparameter combination
        X_aux (Pandas Dataframe): The training dataset for the final model
        X_test (Pandas Dataframe): The target for each sample in the training dataset
        y_aux (Pandas Series): The test dataset for the final model
        y_test (Pandas Series): The target for each sample in the testing dataset
    """

    # Normalising both X_aux and X_test based on X_aux
    X_train, X_test = min_max_scale(X_aux, X_test)
    # y_train, y_test = standard_scale(y_aux, y_test)
    # Creating the time series dataset for the training and testing data
    X_train, y_train = create_timeseries(X_train, y_aux)
    X_test, y_test = create_timeseries(X_test, y_test)

    # Fitting the model on the training dataset with the optimal hyperparameters
    hypermodel.fit(best_hp, model, X_train, y_train)
    # Using the final model to predict the next day closing price for the test sample
    y_pred = model.predict(X_test)

    # Plotting the prediction and ground truth for the test data
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()

    print(compute_accuracy(y_test, y_pred))

    y_train_pred = model.predict(X_train)

    # Plotting the prediction and ground truth for the training data
    plt.plot(y_train)
    plt.plot(y_train_pred)
    plt.show()

    # Saving the model for later reuse in the trading bot
    # model.save("./eth_result_2_2")


# Read the CSV file containing the ETH-USD features for each day
eth_data = pd.read_csv("../../coin_data/eth_data.csv")
# We only train and evaluate the models on data later than 2021-01-01 since this is when ETH-USD started fluctuating massively
eth_data = eth_data.drop(['Date'], axis=1)
# Splitting the CSV file into the feature matrix and the labels
sequence_data, target = eth_data.drop(['Target'], axis=1), eth_data['Target']
# Splitting the feature matrix and labels into the training and testing dataset
X_aux, X_test, y_aux, y_test = chronological_split(sequence_data, target, 0.75)

# Performing Bayesian optimisation to find the best hyperparameters for LSTM neural network to predict ETH-USD price
model, hypermodel, best_hp = lstm_bayes_search(X_aux, y_aux)
# Evaluate the final model on the test data after obtaining the optimal hyperparameters
train_test(model, hypermodel, best_hp, X_aux, X_test, y_aux, y_test)
