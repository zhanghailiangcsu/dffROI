# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:12:13 2022

@author: Administrator
"""

import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers,optimizers,losses,metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense
from tensorflow.keras.layers import Input,concatenate,Flatten
from tensorflow.keras.metrics import BinaryAccuracy
from train.Evaluation_Metrics import TPR,FPR

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        inputA = Input(shape=(256,1))
        inputB = Input(shape=(5,1))
        hp_init = hp.Choice("kernel_initializer", ['tf.keras.initializers.GlorotUniform()',
                                                   'tf.keras.initializers.GlorotNormal()',
                                                   'tf.keras.initializers.HeNormal()',
                                                   'tf.keras.initializers.HeUniform()',
                                                   'tf.keras.initializers.LecunUniform()',
                                                   'tf.keras.initializers.LecunNormal()'])
        for i in range(hp.Int('num_layers', 1, 7)):
            x1 = Conv1D(4*(2**i),5, activation="relu",padding='same',kernel_initializer = eval(hp_init))(inputA)
            x1 = (MaxPooling1D(2))(x1)
        x1 = (Flatten())(x1)
        x1 = Dense(1)(x1)
        x1 = Model(inputs=inputA, outputs=x1)
        
        x2 = (Flatten())(inputB)
        x2 = Model(inputs=inputB, outputs=x2)
        
        x12 = concatenate([x1.output, x2.output])
        x12 = Dense(1, activation="sigmoid")(x12)
        model = Model(inputs=[x1.input, x2.input], outputs=x12)
        hp_opt = hp.Choice("optimizer", ['optimizers.SGD','optimizers.Adam'])
        hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001,0.00005,0.00001])
        model.compile(optimizer = eval(hp_opt+'('+str(hp_learning_rate)+')'),
                      loss=losses.binary_crossentropy,
                      metrics=[BinaryAccuracy(),TPR,FPR])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [50, 100,200,500,1000]),
            epochs = hp.Choice("epochs", [10,20,50,100,200]),
            **kwargs,
        )

tuner = kt.RandomSearch(
    MyHyperModel(),
    objective="val_binary_accuracy",
    max_trials = 2000,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

if __name__ == '__main__':
    
    x_train = np.load('E:/evaluation/train_data/tof/x_train.npy')
    x_ver = np.load('E:/evaluation/train_data/tof/x_ver.npy')
    x_test = np.load('E:/evaluation/train_data/tof/x_test.npy')
    x2_train = np.load('E:/evaluation/train_data/tof/x2_train.npy')
    x2_ver = np.load('E:/evaluation/train_data/tof/x2_ver.npy')
    x2_test = np.load('E:/evaluation/train_data/tof/x2_test.npy')
    y_train = np.load('E:/evaluation/train_data/tof/y_train.npy')
    y_ver = np.load('E:/evaluation/train_data/tof/y_ver.npy')
    y_test = np.load('E:/evaluation/train_data/tof/y_test.npy')
    tuner.search([x_train, x2_train], y_train,validation_data = ([x_ver,x2_ver],y_ver))
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    parameter = best_hps.values












