# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:54:45 2021

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers,losses,metrics
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense
from tensorflow.keras.layers import Input,concatenate,Dropout,Flatten
from tensorflow.keras.metrics import BinaryAccuracy

def TPR(y_true,y_pred): 
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FN)
    return precision

def FPR(y_true,y_pred): 
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=FP/(TN+FP)
    return precision

x_train = np.load('E:/evaluation/train_data/tof/x_train.npy')
x_ver = np.load('E:/evaluation/train_data/tof/x_ver.npy')
x_test = np.load('E:/evaluation/train_data/tof/x_test.npy')
x2_train = np.load('E:/evaluation/train_data/tof/x2_train.npy')
x2_ver = np.load('E:/evaluation/train_data/tof/x2_ver.npy')
x2_test = np.load('E:/evaluation/train_data/tof/x2_test.npy')
y_train = np.load('E:/evaluation/train_data/tof/y_train.npy')
y_ver = np.load('E:/evaluation/train_data/tof/y_ver.npy')
y_test = np.load('E:/evaluation/train_data/tof/y_test.npy')

#peaknoly model
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
# model.summary()

model.compile(optimizer=optimizers.Adam(lr=0.0001),
            loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
history1 = model.fit(x_train, y_train, epochs=50, batch_size=100,
                    validation_data = (x_ver,y_ver))
result = model.evaluate(x_test,y_test)
history_dict1 = history1.history

model.save('D:/code/evaluation/peakonly.h5')

# dffROI model
inputA = Input(shape=(256,1))
inputB = Input(shape=(5,1))
initializer = tf.keras.initializers.LecunUniform()

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
# model.summary()

model2.compile(optimizer=optimizers.Adam(lr = 0.00005),
            loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
history2 = model2.fit([x_train, x2_train], y_train, epochs=200, batch_size=50,
                    validation_data = ([x_ver,x2_ver],y_ver))

result2 = model2.evaluate([x_test,x2_test], y_test)
history_dict2 = history2.history

fig, ax1 = plt.subplots()
plt.tick_params(labelsize=15)
ax2 = ax1.twinx()
l1 = ax1.plot(history_dict2['binary_accuracy'],label = 'accuracy',c = 'r')
l2 = ax1.plot(history_dict2['val_binary_accuracy'],label = 'val_accuracy',c = 'orange')
ax1.set_xlabel("epochs",{'size':15})
ax1.set_ylabel("Accuracy",{'size':15})
ax1.set_ylim([0,1])
l3 = ax2.plot(history_dict2['loss'],label = 'loss',c = 'b')
l4 = ax2.plot(history_dict2['val_loss'],label = 'val_loss',c = 'g')
ax2.set_ylim([0,1])
ax2.set_ylabel("loss",{'size':15})
l = l1+l2+l3+l4
labs = [i.get_label() for i in l]
ax1.legend(l, labs, loc='right',fontsize = 15)
plt.tick_params(labelsize=15)
plt.savefig('D:/paper/train-.tif',dpi=300, bbox_inches='tight')
model2.save('D:/code/evaluation/dffROI.h5')

#handcrafted model
model3 = models.Sequential()
model3.add(layers.Dense(32, activation='relu', input_shape=(5,1)))
model3.add(layers.Dense(16, activation='relu'))
model3.add(layers.Flatten())
model3.add(layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer=optimizers.Adam(lr=0.001),
            loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])

history3 = model3.fit(x2_train, y_train, epochs=20, batch_size=100,
                    validation_data = (x2_ver,y_ver))

history_dict3 = history3.history
result3 = model3.evaluate(x2_test, y_test)
model3.save('D:/code/evaluation/handcrafted.h5')













