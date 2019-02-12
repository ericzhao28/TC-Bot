"""
Created on Feb 11, 2019

@author: Eric Zhao
"""

from .utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np


class Dagger:
    def __init__(self, input_size, hidden_size, output_size):
        model = Sequential()

        model.add(Dense(hidden_size, input_shape=(input_size,)))
        model.add(Activation("relu"))
        model.add(Dropout(0.4))
        model.add(Dense(output_size))
        model.add(Activation("tanh"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=1e-4),
            metrics=["categorical_crossentropy", "accuracy"],
        )

        self.model = model
        self.output_size = output_size

    """ A single batch """

    def train(self, X, Y, batch_size):
        Y = to_categorical(np.array(Y), num_classes=self.output_size, dtype="float32")
        self.model.fit(np.array(X), Y, batch_size=batch_size, nb_epoch=40, shuffle=True)

    """ prediction """

    def predict(self, Xs):
        Ys = self.model.predict(Xs)
        pred_action = np.argmax(Ys)
        return pred_action
