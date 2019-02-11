'''
Created on Feb 11, 2019

@author: Eric Zhao
'''

from .utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np


class Dagger:
    
    def __init__(self, input_size, hidden_size, output_size):
        model = Sequential()

        model.add(Dense(hidden_size, input_shape=input_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(output_size))
        model.add(Activation('tanh'))

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=1e-4),
                      metrics=['mean_squared_error'])


        self.model = model

    """ A single batch """
    def singleBatch(self, X, Y, batch_size):
        self.model.fit(X, Y,
                  batch_size=batch_size,
                  nb_epoch=20,
                  shuffle=True)
    
    """ prediction """
    def predict(self, Xs):
        Ys = self.model.predict(Xs)
        pred_action = np.argmax(Ys)
        return pred_action
