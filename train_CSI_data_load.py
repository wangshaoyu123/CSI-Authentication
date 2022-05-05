# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:56:38 2021
@author: wangshaoyu
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
#import scipy.io
import h5py
#定义CSIDataLoad类，继承Dataset方法，并重写__getitem__()和__len__()方法
# =============================================================================
# class TrainCSIDataLoad(Dataset):
# 	# 初始化函数，得到数据
#     def __init__(self, data_CSI, data_label,transform1,transform2,transform3):
#         self.data = data_CSI
#         self.label = data_label
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.transform3 = transform3
#     # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
#     def __getitem__(self, index):
#         data = self.data[:,:,:,index]
#         #Image.fromarray(data)
#         if 0 <= index < 10000:
#             data = self.transform1(data)
#         elif 10000 <= index < 15000:
#             permutation1 = np.random.permutation(data.shape[0])
#             data = data[permutation1,:,:]
#             data = self.transform1(data)
#         elif 15000 <= index < 20000:
#             permutation2 = np.random.permutation(data.shape[1])
#             data = data[:,permutation2,:]
#             data = self.transform1(data)
# # =============================================================================
# #         elif 20000 <= index < 21000:
# #             permutation3 = np.random.permutation(data.shape[0])
# #             permutation4 = np.random.permutation(data.shape[1])
# #             data = data[permutation3,:,:]
# #             data = data[:,permutation4,:]
# #             data = self.transform2(data)
# # =============================================================================
#         elif 20000 <= index < 30000:
#             data = self.transform2(data)
#         else:
#             data = self.transform3(data)
#         labels = self.label[index]
#         return data, labels
#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，CSIDataLoad会一脸懵逼
#     def __len__(self):
#         #return len(self.data)
#         return np.size((self.data),3)
# =============================================================================

class TrainCSIDataLoad(Dataset):
	# 初始化函数，得到数据
    # def __init__(self, data_CSI, data_label,transform1,transform2,transform3):
    #     self.data = data_CSI
    #     self.label = data_label
    #     self.transform1 = transform1
    #     self.transform2 = transform2
    #     self.transform3 = transform3
    # # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    # def __getitem__(self, index):
    #     data = self.data[:,:,:,index]
    #     if 14000 <= index < 17000:
    #         permutation1 = np.random.permutation(data.shape[0])
    #         data = data[permutation1,:,:]
    #         data = self.transform1(data)
    #     elif 17000 <= index < 20000:
    #         permutation2 = np.random.permutation(data.shape[1])
    #         data = data[:,permutation2,:]
    #         data = self.transform1(data)
    #     elif 20000 <= index < 24000:
    #         permutation3 = np.random.permutation(data.shape[0])
    #         permutation4 = np.random.permutation(data.shape[1])
    #         data = data[permutation3,:,:]
    #         data = data[:,permutation4,:]
    #         data = self.transform1(data)
    #     elif 24000 <= index < 31000:  
    #         data = self.transform2(data)
    #     else:  
    #         data = self.transform3(data)
    #     labels = self.label[index]
    #     return data, labels
    def __init__(self, data_CSI, data_label,transform1,transform2,transform3,transform4):
        self.data = data_CSI
        self.label = data_label
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[:,:,:,index]
        if 14000 <= index < 17000:
            permutation1 = np.random.permutation(data.shape[0]) #交换行
            data = data[permutation1,:,:]
            data = self.transform1(data)
        elif 17000 <= index < 20000:
            data = self.transform2(data)
        elif 20000 <= index < 23000:
            permutation2 = np.random.permutation(data.shape[1]) #交换列
            data = data[:,permutation2,:]
            data = self.transform1(data)
        elif 23000 <= index < 30000:
            permutation3 = np.random.permutation(data.shape[0])  #交换行和列
            permutation4 = np.random.permutation(data.shape[1])
            data = data[permutation3,:,:]
            data = data[:,permutation4,:]
            data = self.transform1(data)
        elif 30000 <= index < 33000:  
            data = self.transform3(data)
        elif 33000 <= index < 40000:
            data = self.transform4(data)
        else: 
            data = self.transform1(data)
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，CSIDataLoad会一脸懵逼
    def __len__(self):
        #return len(self.data)
        return np.size((self.data),3)
        
        
         
        
        
        
        









        