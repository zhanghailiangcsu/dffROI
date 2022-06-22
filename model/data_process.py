# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:46:21 2021

@author: Administrator
"""
import json
import os
from sklearn.preprocessing import normalize
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

def process(path):
    path2 = os.listdir(path)
    rt = []
    label = []
    intensity = []
    for i in path2:
        p = path+'/'+i
        with open(p) as f:
            df = json.load(f)
            label.append(df['label'])
            intensity.append(df['intensity'])
            rt.append(df['rt'])
    return rt,label,intensity

def getdata(path):
    pathlist = os.listdir(path)
    rt = []
    label = []
    intensity = []
    for p in pathlist:
        p = path+'/'+p
        rt1,label1,intensity1 = process(p)
        rt = rt+rt1
        label = label+label1
        intensity = intensity+intensity1
    return rt,label,intensity

def data_resize(data, new_length):
    f = interp1d(np.linspace(0, new_length-1, num=len(data), endpoint=True), data, kind='cubic')
    x_new = np.linspace(0, new_length-1, num=new_length, endpoint=True)
    y_new = f(x_new).reshape(new_length,1)
    y_new = normalize(y_new, axis=0, norm='max')
    return y_new

def smoothsim(intensity):
    xic2 = np.array([])
    for i in range(2,len(intensity)-2):
        xic2 = np.append(xic2,np.mean((intensity[i-2],intensity[i-1],
                                       intensity[i],intensity[i+1],intensity[i+2])))
    xic2 = np.hstack((xic2[0],xic2))
    xic2 = np.hstack((xic2[0],xic2))
    xic2 = np.hstack((xic2,xic2[-1]))
    xic2 = np.hstack((xic2,xic2[-1]))
    p = abs(pearsonr(intensity, xic2)[0])
    return p

def zig(intensity):
    epi = max(intensity)-min(intensity)
    s = 0
    for i in range(1,len(intensity)-1):
        s = s + (2*intensity[i]-intensity[i-1]-intensity[i+1])**2
    z = s/(len(intensity)*epi**2)
    return z

def smooth(intensity):
    xic2 = np.array([])
    for i in range(2,len(intensity)-2):
        xic2 = np.append(xic2,np.mean((intensity[i-2],intensity[i-1],
                                       intensity[i],intensity[i+1],intensity[i+2])))
    xic2 = np.hstack((xic2[0],xic2))
    xic2 = np.hstack((xic2[0],xic2))
    xic2 = np.hstack((xic2,xic2[-1]))
    xic2 = np.hstack((xic2,xic2[-1]))
    return xic2

def change(intensity):
    zig0 = zig(intensity)
    zig1 = zig(smooth(intensity))
    return zig0-zig1

def cal_snr(intensity):
    m = max(intensity)
    inten = [i for i in intensity if i !=0]
    b = np.mean([inten[0:2]+inten[-3:-1]])
    return m/b

def getfeature(intensity):
    fea = []
    snr = np.array([])
    maxinten = np.array([])
    for i in range(0,len(intensity)):
        feature = np.array([])
        feature = np.append(feature,smoothsim(intensity[i]))
        feature = np.append(feature,zig(intensity[i]))
        feature = np.append(feature,change(intensity[i]))
        snr = np.append(snr,cal_snr(intensity[i]))
        maxinten = np.append(maxinten,max(intensity[i]))
        fea.append(feature)
    snr = snr.reshape(len(snr),1)
    snr = normalize(snr, axis=0, norm='max')
    maxinten = maxinten.reshape(len(maxinten),1)
    maxinten = normalize(maxinten, axis=0, norm='max')
    for i in range(0,len(fea)):
        fea[i] = np.append(fea[i],snr[i])
        fea[i] = np.append(fea[i],maxinten[i])
        fea[i] = fea[i].reshape(len(fea[i]),1)
    return fea

def randomize(dataset, dataset2,label):

    permutation = np.random.permutation(len(label))
    dataset = dataset[permutation,:]
    dataset2 = dataset2[permutation,:]
    label = label[permutation]
    return dataset,  dataset2,label

if __name__ == "__main__":
    
    path = 'E:/peakonly/data/tof'
    rt,label,intensity = getdata(path)
    intensity2 = [data_resize(i,256) for i in intensity]
    feature = getfeature(intensity)
    intensity2 = np.array(intensity2)
    label2 =np.array(label).reshape(len(label),1)
    feature2 = np.array(feature)
    intensity3,fea3,label3 = randomize(intensity2, feature2,label2)
    
    x_train = intensity3[0:1800,:,:]
    x_ver = intensity3[1800:2050,:,:]
    x_test = intensity3[2050:,:,:]
    x2_train = fea3[0:1800,:,:]
    x2_ver = fea3[1800:2050,:,:]
    x2_test = fea3[2050:,:,:]
    y_train = label3[0:1800,:]
    y_ver = label3[1800:2050,:]
    y_test = label3[2050:,:]
    np.save('E:/evaluation/train_data/tof/x_train.npy',x_train)
    np.save('E:/evaluation/train_data/tof/x_ver.npy',x_ver)
    np.save('E:/evaluation/train_data/tof/x_test.npy',x_test)
    np.save('E:/evaluation/train_data/tof/x2_train.npy',x2_train)
    np.save('E:/evaluation/train_data/tof/x2_ver.npy',x2_ver)
    np.save('E:/evaluation/train_data/tof/x2_test.npy',x2_test)
    np.save('E:/evaluation/train_data/tof/y_train.npy',y_train)
    np.save('E:/evaluation/train_data/tof/y_ver.npy',y_ver)
    np.save('E:/evaluation/train_data/tof/y_test.npy',y_test)
    
    path = 'E:/peakonly/data/icr'
    rt,label,intensity = getdata(path)
    intensity2 = [data_resize(i,256) for i in intensity]
    feature = getfeature(intensity)
    intensity2 = np.array(intensity2)
    label2 =np.array(label).reshape(len(label),1)
    feature2 = np.array(feature)
    np.save('E:/evaluation/train_data/icr/intensity2.npy',intensity2)
    np.save('E:/evaluation/train_data/icr/feature2.npy',feature2)
    np.save('E:/evaluation/train_data/icr/label2.npy',label2)
    
    path = 'E:/peakonly/data/orbitrap'
    rt,label,intensity = getdata(path)
    intensity2 = [data_resize(i,256) for i in intensity]
    feature = getfeature(intensity)
    intensity2 = np.array(intensity2)
    label2 =np.array(label).reshape(len(label),1)
    feature2 = np.array(feature)
    np.save('E:/evaluation/train_data/orbitrap/intensity2.npy',intensity2)
    np.save('E:/evaluation/train_data/orbitrap/feature2.npy',feature2)
    np.save('E:/evaluation/train_data/orbitrap/label2.npy',label2)
    
    path = 'E:/peakonly/data/gc'
    rt,label,intensity = getdata(path)
    intensity2 = [data_resize(i,256) for i in intensity]
    feature = getfeature(intensity)
    intensity2 = np.array(intensity2)
    label2 =np.array(label).reshape(len(label),1)
    feature2 = np.array(feature)
    np.save('E:/evaluation/train_data/gc/intensity2.npy',intensity2)
    np.save('E:/evaluation/train_data/gc/feature2.npy',feature2)
    np.save('E:/evaluation/train_data/gc/label2.npy',label2)
    
    
    
    































