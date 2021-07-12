#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
#
from sys import exit
import datetime
import os
import torch
import torch.optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#
import functional
import model_define
import data_pre_with_mp
import run

# constant
USE_GPU = False
DEBUG = True
CH1_VOLTAGE_LEN = 161
CRITERION = nn.MSELoss()

if __name__ == '__main__':
    ###################################################################################################################
    # prepare data set
    # data file path
    n_epochs = 256
    batch_size = 256
    data_file_path = 'D:\\workspace\\PycharmProjects\\battery_dataset'
    file_organize_type = ('2600P-01_DataSet_Balance', )  # ('2600P-01_DataSet_Balance', '2600P-01_DataSet_UnBalance')
    data_process_type = ('npy_standardized', )  # ('npy_standardized', 'npy_unstandardized')

    # data set profile
    dataset_profile_list = []
    for og_type in file_organize_type:
        temp_og_path = os.path.join(data_file_path, og_type)
        assert os.path.exists(temp_og_path)
        for dp_type in data_process_type:
            temp_dataset_type = f'{og_type}_{dp_type}'
            temp_npy_path = os.path.join(temp_og_path, dp_type)
            assert os.path.exists(temp_npy_path)
            temp_dataset_name = f'data_set_{temp_dataset_type}'
            temp_dataset_path = os.path.join(temp_og_path, temp_dataset_name)
            if not os.path.exists(temp_dataset_path):
                os.mkdir(temp_dataset_path)
                os.mkdir(temp_dataset_path + '\\train')
                os.mkdir(temp_dataset_path + '\\val')
                os.mkdir(temp_dataset_path + '\\test')
                data_pre_with_mp.DataProcessor.data_divide(temp_npy_path, temp_dataset_path)
            # creat data set
            train_path = os.path.join(temp_dataset_path + '\\train')
            val_path = os.path.join(temp_dataset_path + '\\val')
            test_path = os.path.join(temp_dataset_path + '\\test')
            
            train_dataset, n_train_batch = data_pre_with_mp.DataSetCreator.creat_dataset(train_path, bsz=batch_size)
            val_dataset, n_val_batch = data_pre_with_mp.DataSetCreator.creat_dataset(val_path, bsz=1)
            test_dataset, n_test_batch = data_pre_with_mp.DataSetCreator.creat_dataset(test_path, bsz=1)

            temp_dataset_profile_dic = {
                'dataset_type': temp_dataset_type,
                'train_data_set': (train_dataset, n_train_batch),
                'val_data_set': (val_dataset, n_val_batch),
                'test_data_set': (test_dataset, n_test_batch),
            }
            dataset_profile_list.append(temp_dataset_profile_dic)
    ###################################################################################################################
    # Model Setup and init
    #
    # PureMLP setup
    #  select device
    if USE_GPU:
        device = functional.try_gpu()
    else:
        device = 'cpu'

    model_init_para = {
        'in_dim': CH1_VOLTAGE_LEN
    }
    # init model
    model = model_define.PureMLP(**model_init_para)
    model_name = model.model_name
    model.to(device=device)
    model_dict = {
        'model_name': model_name,
        'model_init_para': model_init_para,
        'model': model,
        'device': device
    }
    ###################################################################################################################
    # optimizer set
    type_optimizer = 'SGD'  # SGD or Adam
    init_lr = 0.001
    if type_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9),
        optimizer = optimizer[0]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    elif type_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.998), eps=1e-09)
        scheduler = None
    else:
        raise ValueError('Optimizer Type Error.')
    # form into dict
    op_dict = {
        'init_lr': init_lr,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    ###################################################################################################################
    # setup mode
    log_dir_base = 'D:\\workspace\\PycharmProjects\\model_run'

    if DEBUG:
        log_dir = log_dir_base + '\\DEBUG\\' + f'\\{model_name}\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = log_dir_base + f'\\{model_name}\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard summary define
    with SummaryWriter(log_dir=log_dir) as writer:

        # model save path and name
        model_path = log_dir
        print(f'Model Save Path: {model_path}')

        # dataset loop
        for curr_dataset in dataset_profile_list:

            curr_dataset_name = curr_dataset['dataset_type']
            curr_dataset_dic = {
                'train_data_set': curr_dataset['train_data_set'][0],
                'num_of_train_batch': curr_dataset['train_data_set'][1],
                'val_data_set': curr_dataset['val_data_set'][0],
                'num_of_val_batch': curr_dataset['val_data_set'][1],
                'test_data_set': curr_dataset['test_data_set'][0],
                'num_of_test_batch': curr_dataset['test_data_set'][1]
            }
            #
            curr_dataset_model_path = os.path.join(model_path, curr_dataset_name)
            if not os.path.exists(curr_dataset_model_path): 
                os.mkdir(curr_dataset_model_path)

            #
            m_run = run.Run(model_dict['model_name'],
                            model_dict['model'],
                            op_dict['optimizer'],
                            CRITERION,
                            curr_dataset_dic,
                            scheduler=op_dict['scheduler'],
                            device=model_dict['device'],
                            writer=writer)
            m_run.train_model(n_epochs)
            # test
            m_run.test_model()
        exit(0)
