# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:15:13 2022

@author: Administrator
"""
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers,losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense
from tensorflow.keras.layers import Input,concatenate,Flatten
from tensorflow.keras.metrics import BinaryAccuracy
from train.Evaluation_Metrics import TPR,FPR

def PeakonlyModel():
    model = models.Sequential()
    model.add(layers.Conv1D(8,5,activation='relu',padding='same',input_shape=(256,1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(16,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(256,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(512,5, activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=0.0001),
            loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    return model

def dffROIModel():
    inputA = Input(shape=(256,1))
    inputB = Input(shape=(5,1))
    initializer = tf.keras.initializers.GlorotUniform()
    
    x1 = Conv1D(8,5, activation="relu",padding='same',kernel_initializer = initializer)(inputA)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Conv1D(16,5, activation="relu",padding='same',kernel_initializer = initializer))(x1)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Conv1D(32,5, activation="relu",padding='same',kernel_initializer = initializer))(x1)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Conv1D(64,5, activation="relu",padding='same',kernel_initializer = initializer))(x1)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Conv1D(128,5, activation="relu",padding='same',kernel_initializer = initializer))(x1)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Conv1D(256,5, activation="relu",padding='same',kernel_initializer = initializer))(x1)
    x1 = (MaxPooling1D(2))(x1)
    x1 = (Flatten())(x1)
    x1 = Dense(1)(x1)
    x1 = Model(inputs=inputA, outputs=x1)
    
    x2 = (Flatten())(inputB)
    x2 = Model(inputs=inputB, outputs=x2)
    
    x12 = concatenate([x1.output, x2.output])
    x12 = Dense(1, activation="sigmoid")(x12)
    model2 = Model(inputs=[x1.input, x2.input], outputs=x12)
    model2.compile(optimizer=optimizers.Adam(lr = 0.0001),
            loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    return model2

def HandcraftedModel():
    model3 = models.Sequential()
    model3.add(layers.Dense(32, activation='relu', input_shape=(5,1)))
    model3.add(layers.Dense(16, activation='relu'))
    model3.add(layers.Flatten())
    model3.add(layers.Dense(1, activation='sigmoid'))
    
    model3.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[BinaryAccuracy(),TPR,FPR])
    return model3
















