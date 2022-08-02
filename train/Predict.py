# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:36:31 2022

@author: Administrator
"""
import numpy as np
from train.Evaluation_Metrics import TPR,FPR
from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy

def predict(model,x_test,x2_test, y_test):
    result = model.evaluate([x_test,x2_test], y_test)
    return result

if __name__ =='__main__':
    
    x_test = np.load('E:/evaluation/train_data/tof/x_test.npy')
    x2_test = np.load('E:/evaluation/train_data/tof/x2_test.npy')
    y_test = np.load('E:/evaluation/train_data/tof/y_test.npy')
    
    _custom_objects = {"TPR":TPR, "FPR":FPR}
    dffROI_model = tf.keras.models.load_model('dffROI.h5',custom_objects=_custom_objects)
    dffROI_model.compile(loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    result = predict(dffROI_model,x_test,x2_test, y_test)
    
    
    