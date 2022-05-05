# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:56:38 2021
@author: wangshaoyu
"""
import torch
import numpy as np
#from PIL import Image
from torch.utils.data import Dataset
#import scipy.io
import h5py
#定义CSIDataLoad类，继承Dataset方法，并重写__getitem__()和__len__()方法
class TestCSIDataLoad(Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_CSI, data_label,transform):
        self.data = data_CSI
        self.label = data_label
        self.transform = transform
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[:,:,:,index]
        data = self.transform(data)
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，CSIDataLoad会一脸懵逼
    def __len__(self):
        #return len(self.data)
        return np.size((self.data),3)

# 随机生成数据，大小为10 * 20列
#source_data = torch.rand(10,20,10,100)
#print(source_data[:,:,:,99])
# 随机生成标签，大小为10 * 1列
#source_label = np.random.randint(0,2,(10, 1))
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
#CSIData = scipy.io.loadmat('channel(10000).mat') # 读取mat文件
# =============================================================================
# CSIDataFileLoad = h5py.File('channel(10000).mat','r') # 读取mat文件
# #print(CSIData1.keys())
# CSIData = np.transpose(CSIDataFileLoad['channel'][:]) #转置后才能和channel的维度一致
# CSILable = np.zeros(10000)
# for i in range(0,100):
#     CSILable[0+100*i:100+100*i] = i*np.ones(100)
# #print(np.size(CSIData,3))
# torch_data = CSIDataLoad(CSIData, CSILable)
# =============================================================================
        
        
        
        
        
        
        









        