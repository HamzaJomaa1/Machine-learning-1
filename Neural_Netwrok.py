#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM

class NeuralNetwork:
    @staticmethod
    def build_feedforward_model(input_shape):
        model = Sequential([
            Dense(32, activation='relu', input_shape=input_shape),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def build_cnn_model(input_shape):
        model = Sequential([
            Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=input_shape),
            Flatten(),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def build_rnn_model(input_shape):
        model = Sequential([
            LSTM(16, input_shape=input_shape),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


# In[ ]:




