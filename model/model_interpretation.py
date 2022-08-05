# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 10:48:53 2021

@author: Administrator
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

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

def data_random(intensity2,fea2,feature = 0):
    permutation = np.random.permutation(intensity2.shape[0])
    if feature == 0:
        intensity2 = intensity2[permutation,:,:]
    if feature == 1:
        fea = fea2[:,0,:]
        fea = fea[permutation,:]
        fea2[:,0,:] = fea
    if feature == 2:
        fea = fea2[:,1,:]
        fea = fea[permutation,:]
        fea2[:,1,:] = fea
    if feature == 3:
        fea = fea2[:,2,:]
        fea = fea[permutation,:]
        fea2[:,2,:] = fea
    if feature == 4:
        fea = fea2[:,3,:]
        fea = fea[permutation,:]
        fea2[:,3,:] = fea
    if feature == 5:
        fea = fea2[:,4,:]
        fea = fea[permutation,:]
        fea2[:,4,:] = fea
    return intensity2,fea2

if __name__ == '__main__':
    
    intensity2 = np.load('E:/evaluation/train_data/intensity2.npy')
    fea2 = np.load('E:/evaluation/train_data/feature2.npy')
    label2 = np.load('E:/evaluation/train_data/label2.npy')
    label2 = label2.reshape(len(label2),1)

    _custom_objects = {"TPR":TPR, "FPR":FPR}
    dffROI_model = load_model('D:/code/evaluation/dffROI.h5',custom_objects=_custom_objects)
    result = dffROI_model.evaluate([intensity2,fea2], label2)
    result1 = []
    for i in range(10):
        x_test,x2_test = data_random(intensity2,fea2,feature = 0)
        r = dffROI_model.evaluate([intensity2,fea2], label2)
        result1.append(r)
    fi = np.mean(np.array([[i[1] for i in result1]]))














