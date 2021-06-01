#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import sys
import os
import data_loader
import torch.utils.data as t_u_data
import numpy as np

#  select which data to be use.
stage_list = ('Static', 'Charge #1', 'Charge #2', 'Charge #3', 'Discharge',  'Charge #4')
param_list = ('time', 'voltage', 'current', 'capacity')
data_file_type = ('xlsx', 'pickle')
running_mode = ('running', 'debug')
#  1 - use, 0 - no use
param_mode_dic = {
                 'Static': ['Form-OCV #1', ],
                 'Charge #1': ['time', 'voltage'],
                 'Charge #2': [],
                 'Charge #3': [],
                 'Discharge': [],
                 'Charge #4': []
                 }
epochs = 10
batch_size = 32
num_of_worker = 0
# Fold Structure
# data\2600P-01_DataSet
# |
# |-- pickle
# |-- xlsx
# |-- dataset
#     |
#     |--test
#     |--train
#     |--val
#
#  path of dataset
cells_data_path = '.\\data\\2600P-01_DataSet\\pickle'  # Load pickle data
cells_divided_data_path = '.\\data\\2600P-01_DataSet\\dataset'  # path store divided dataset

if __name__ == '__main__':
    # Load data from folds
    m_dataLoader = data_loader.DataLoader(cells_data_path, cells_divided_data_path, param_mode_dic)
    train_path = '.\\data\\2600P-01_DataSet\\dataset\\train'
    cell_data_set = data_loader.CellDataSet(train_path)
    batch_data = t_u_data.DataLoader(cell_data_set, batch_size=batch_size, shuffle=True)
    example = np.load('.\\data\\2600P-01_DataSet\\dataset\\train\\210301-1_C000000397_1.npy')
    data_arr_batch = np.zeros((batch_size, example.shape[0]))
    for data_list in batch_data:
        data_arr_batch = np.zeros((batch_size, example.shape[0]))
        for i, data in enumerate(iter(data_list)):
            data_path = os.path.join(train_path, data)
            arr = np.load(data_path)
            arr_T = arr.T
            data_arr_batch[i, :] = arr_T
sys.exit(0)
