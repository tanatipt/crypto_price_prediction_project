import tensorflow as tf
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.losses import mean_squared_error
from tensorflow.keras.utils import get_custom_objects
import keras_tuner as kt

get_custom_objects().clear()
get_custom_objects().update({'leaky_relu': tf.keras.layers.LeakyReLU()})


def root_mean_squared_error(y_true, y_pred):
    """

    Defining the RMSE loss function for the Keras neural network. This loss function was chosen for Cryptocurrency price prediction since
    the dataset is extremely noisy so RMSE reduces the magitude that outliers have on the loss ; hence, the updating of the weight and biases
    of the neural network

    Args:
        y_true (Numpy Array): An array of true labels for each sample
        y_pred (Numpy Array): An array of predicted labels for each sample

    Returns:
        Tensor: The RMSE between true and predicted labels
    """
    return K.sqrt(mean_squared_error(y_true, y_pred))


class MyHyperModel(kt.HyperModel):
    """

    A Hypermodel class that inherits from the Keras Hypermodel class. This class allows us to build
    an LSTM neural network using the build function as welll as fitting it with some dataset.

    Args:
        kt (Keras HyperModel): A HyperModel object
    """

    def build(self, hp):
        """

        This function builds a specific architecture of the LSTM neural network for ETH-USD price prediction as well as 
        defining the hyperparameters value to be evaluated by the Bayesian optimisation search. The neural network consists of 
        2 hidden LSTM cells and 2 dense layers where each hidden layer is immediately followed by a dropout layer. The RMSE was used
        as the loss function as there are many outliers in the dataset so the function reduces the effect of outliers on the loss; hence, 
        reducing the effect of weights/biases update caused by outliers.

        Args:
            hp (Keras Hyperparameters):  A HyperParameter instance that allows us to define the hyperparameters durig model building

        Returns:
            Keras Model : A Keras LSTM neural network model that has been compiled
        """

        # Defining the number of hidden units in the 1st and 2nd LSTM hidden cell
        hp_lstm_1 = hp.Int('lstm_1', min_value=8, max_value=128,
                           step=2, sampling="log")
        hp_lstm_2 = hp.Int('lstm_2', min_value=8, max_value=128,
                           step=2, sampling="log")

        # Defining the number of hidden units in the 1st and 2nd Dense hidden layers
        hp_dense_1 = hp.Int('dense_1', min_value=8,
                            max_value=128, step=2, sampling="log")
        hp_dense_2 = hp.Int('dense_2', min_value=8,
                            max_value=128, step=2, sampling="log")

        # Defining the dropout rate associated with each dropout layer
        dropout_rate = hp.Float('dropout', min_value=0.0,
                                max_value=0.3, step=0.1)
        # Specifying the learning rate for the Adam optimizer of the model
        learning_rate = hp.Float(
            'learning_rate', min_value=0.001, max_value=0.1, step=10, sampling="log")
        # Specifying the weight initialisation method for each hidden lyaer
        weight_initialiser = hp.Choice(
            'weight_initialiser', values=['random_normal', 'glorot_normal', 'he_normal'])
        # Specifying the activation function to use for the dense layers. By default, the activation function of LSTM layer is tanh
        activation_function = hp.Choice(
            'activation', values=['relu', 'elu', 'leaky_relu'])

        input = Input(shape=(7, 25))

        # Defining the 1st and 2nd LSTM layer with varying number of hidden units
        lstm_1 = LSTM(units=hp_lstm_1,
                      kernel_initializer=weight_initialiser,  return_sequences=True)(input)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        lstm_2 = LSTM(units=hp_lstm_2,
                      kernel_initializer=weight_initialiser)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(lstm_2)

        # Defining the 1st and 2nd Dense layer with varying number of hidden units
        dense_1 = Dense(units=hp_dense_1,
                        activation=activation_function, kernel_initializer=weight_initialiser)(dropout_2)
        dropout_3 = Dropout(rate=dropout_rate)(dense_1)
        dense_2 = Dense(units=hp_dense_2,
                        activation=activation_function, kernel_initializer=weight_initialiser)(dropout_3)
        dropout_4 = Dropout(rate=dropout_rate)(dense_2)

        # Since price prediction is a regression task, we use a single node in the output layer and use the linear activation function
        output_reg = Dense(2,  kernel_initializer=weight_initialiser,
                           activation="softmax")(dropout_4)

        # Configuring the Adam optimizer object
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = Model(inputs=input, outputs=output_reg)

        # Compiling the model to use the RMSE as the loss function and Adam as the optimizer
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """

        Performing a fit on the given LSTM neural network with a specified number of epochs and batch size. The epochs and batch size are 
        the hyperparameters that we have defined during the fitting process. The training dataset and labels will be used to fit the model.

        Args:
            hp (Keras Hyperparameters): A HyperParameter instance that allows us to define the hyperparameters durig model fitting
            model (Keras Model): A specific LSTM neural network architecture that will be fitted

        Returns:
            Keras History: A History object containing information for each epoch during fitting
        """

        return model.fit(
            *args,
            epochs=hp.Int("epochs", min_value=50, max_value=250, step=50),
            batch_size=hp.Int("batch", min_value=8, max_value=32, step=8),
            ** kwargs,
        )
