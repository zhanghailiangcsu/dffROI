# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:18:40 2022

@author: Administrator
"""
from model.Build_Model import dffROIModel
import numpy as np


def train(model,x_train, x2_train, y_train,
          x_ver,x2_ver, y_ver,
          epochs,batch_size):
    history = model.fit([x_train, x2_train], y_train, epochs=epochs, 
                        batch_size=batch_size,
                        validation_data = ([x_ver,x2_ver],y_ver))
    history_dict = history.history
    return model,history_dict

if __name__ =='__main__':
    
    model = dffROIModel()
    x_train = np.load('E:/evaluation/train_data/tof/x_train.npy')
    x_ver = np.load('E:/evaluation/train_data/tof/x_ver.npy')
    x2_train = np.load('E:/evaluation/train_data/tof/x2_train.npy')
    x2_ver = np.load('E:/evaluation/train_data/tof/x2_ver.npy')
    y_train = np.load('E:/evaluation/train_data/tof/y_train.npy')
    y_ver = np.load('E:/evaluation/train_data/tof/y_ver.npy')
    epochs = 50
    batch_size = 50
    dffROI_model,history_dict = train(model,x_train, x2_train, y_train,x_ver,x2_ver, y_ver,
          epochs,batch_size)
    dffROI_model.save('dffROI.h5')
