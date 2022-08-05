# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:36:39 2021

@author: Administrator
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses,metrics
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


if __name__ == '__main__':
    
    _custom_objects = {"TPR":TPR, "FPR":FPR}
    dffROI_model = tf.keras.models.load_model('D:/code/evaluation/dffROI.h5',custom_objects=_custom_objects)
    dffROI_model.compile(loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    peakonly_model = tf.keras.models.load_model('D:/code/evaluation/peakonly.h5',custom_objects=_custom_objects)
    peakonly_model.compile(loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    handcrafted_model = tf.keras.models.load_model('D:/code/evaluation/handcrafted.h5',custom_objects=_custom_objects)
    handcrafted_model.compile(loss=losses.binary_crossentropy,
            metrics=[BinaryAccuracy(),TPR,FPR])
    
    
    #test on ICR
    intensity2 = np.load('E:/evaluation/train_data/icr/intensity2.npy')
    feature2 = np.load('E:/evaluation/train_data/icr/feature2.npy')
    label2 = np.load('E:/evaluation/train_data/icr/label2.npy')
    result1 = peakonly_model.evaluate(intensity2, label2)
    result2 = dffROI_model.evaluate([intensity2,feature2], label2)
    result3 = handcrafted_model.evaluate(feature2, label2)
    
    #test on orbitrap
    intensity2 = np.load('E:/evaluation/train_data/orbitrap/intensity2.npy')
    feature2 = np.load('E:/evaluation/train_data/orbitrap/feature2.npy')
    label2 = np.load('E:/evaluation/train_data/orbitrap/label2.npy')
    result4 = peakonly_model.evaluate(intensity2, label2)
    result5 = dffROI_model.evaluate([intensity2,feature2], label2)
    result6 = handcrafted_model.evaluate(feature2, label2)
    
    #test on gc
    intensity2 = np.load('E:/evaluation/train_data/gc/intensity2.npy')
    feature2 = np.load('E:/evaluation/train_data/gc/feature2.npy')
    label2 = np.load('E:/evaluation/train_data/gc/label2.npy')
    result7 = peakonly_model.evaluate(intensity2, label2)
    result8 = dffROI_model.evaluate([intensity2,feature2], label2)
    result9 = handcrafted_model.evaluate(feature2, label2)
