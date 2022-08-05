# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:32:33 2021

@author: Administrator
"""
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from tensorflow.keras.models import load_model
from model.data_process import getfeature,data_resize
from train.Evaluation_Metrics import TPR,FPR

def data_transform(data):
    data = list(data)
    data[0] = list(data[0])
    data[1] = np.array(data[1])
    data[2] = list(data[2])
    data[2] = [np.array(i) for i in data[2]]
    data[3] = list(data[3])
    return data

def get_kpic_roi(pics):
    rois = []
    for pic in pics:
        p = pic[2]
        roi = [list(i[:,1]) for i in p]
        rois.append(roi)
    return rois

def model_process(model,pics):
    result = []
    roi= pics[0]
    for roi in pics:
        intensity = [data_resize(i,256) for i in roi]
        intensity2 = np.array(intensity)
        feature = getfeature(roi)
        feature2 = np.array(feature)
        r = np.round(model.predict([intensity2,feature2]))
        r= [index+1 for index,value in enumerate(r) if value == 0]
        r = r[::-1]
        result.append(r)
    return result

if __name__ == '__main__':
    numpy2ri.activate()
    robjects.r('''source('kpic2/kpic_process.R')''')
    kpic_pic = robjects.globalenv['kpic_pic']
    kpic_pic_set = robjects.globalenv['kpic_pic_set']
    kpic_pic_getpeak = robjects.globalenv['kpic_pic_getpeak']
    kpic_group = robjects.globalenv['kpic_group']
    kpic_iso = robjects.globalenv['kpic_iso']
    kpic_mat = robjects.globalenv['kpic_mat']
    kpic_fill = robjects.globalenv['kpic_fill']
    kpic_pattern = robjects.globalenv['kpic_pattern']
    kpic_select = robjects.globalenv['kpic_select']
    
    filename = 'E:/evaluation/MTBLS120/posdemo2'
    pics = kpic_pic_set(filename,level = 2000)
    
    pics2 = list(pics)
    pics3 = [data_transform(i) for i in pics2]
    pics4 = get_kpic_roi(pics3)
    _custom_objects = {"TPR":TPR, "FPR":FPR}
    dffROI_model = load_model('D:/code/evaluation/dffROI.h5',custom_objects=_custom_objects)
    result = model_process(dffROI_model,pics4)
        
    del_list = []
    for d in range(len(result)):
        d0 = kpic_select(result[d],pics2[d])
        del_list.append(d0)
    del_rlist = robjects.ListVector([(str(i), x) for i, x in enumerate(del_list)])
    
    del_rlist = kpic_pic_getpeak(del_rlist)
    
    groups_align = kpic_group(del_rlist)
    groups_align = kpic_iso(groups_align)
    data = kpic_mat(groups_align)
    data = kpic_fill(data)
    data = list(data)
    data[3] = np.array(data[3])
    df = np.array(data[3])
    np.save('E:/evaluation/MTBLS120/pos2.npy',df)
    
    
    
    
    

    
    
    
    
    





