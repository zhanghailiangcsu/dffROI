# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:31:42 2022

@author: Administrator
"""
from model.Build_Model import dffROIModel
from train.Evaluation_Metrics import TPR,FPR
from train.Train_Model import train
import numpy as np
import os
os.getcwd()

if __name__ =='__main__':
    
    sb =0