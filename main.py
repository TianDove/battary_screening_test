#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
#
from sys import exit
import time
import datetime
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

#
import functional
import model_define
import data_loader
import run

# constant
USE_GPU = True
DEBUG = False

if __name__ == '__main__':
    # tensorboard summary define
    if DEBUG:
        log_dir = '.\\runs\\DEBUG\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = '.\\runs\\MyModel\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with SummaryWriter(log_dir=log_dir) as writer:
        #  select which data to be use.
        # stage_list = ('Static', 'Charge #1', 'Charge #2', 'Charge #3', 'Discharge', 'Charge #4')
        # param_list = ('time', 'voltage', 'current', 'capacity')
        # data_file_type = ('xlsx', 'pickle')
        # running_mode = ('running', 'debug')
        #  1 - use, 0 - no use
        """param_mode_dic = {
            'Static': ['Form-OCV #1', ],
            'Charge #1': ['time', 'voltage'],
            'Charge #2': [],
            'Charge #3': [],
            'Discharge': [],
            'Charge #4': []
        }"""
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
        # cells_data_path = '.\\data\\2600P-01_DataSet\\pickle'  # Load pickle data
        # cells_divided_data_path = '.\\data\\2600P-01_DataSet\\dataset'  # path store divided dataset

        # file path
        train_path = '.\\data\\2600P-01_DataSet\\dataset\\train'
        val_path = '.\\data\\2600P-01_DataSet\\dataset\\val'
        # test path
        test_path = '.\\data\\2600P-01_DataSet\\dataset\\test'
        # model save path and name
        model_path = log_dir
        print(f'Model Save Path: {model_path}')

        # set up hyper parameter
        d_model = 8
        is_overlap = False
        t_step = 8
        tokenize_tup = (d_model, is_overlap, t_step)
        epochs = 250
        batch_size = 128
        num_of_worker = 2

        #
        tokenizer = functional.Tokenizer(tokenize_tup)
        tokenizer.calculate_expamle_detoken_len(train_path)

        # models_para_dic
        models_para_dic = {
            'net1': {},

        }
        # model_para_dic
        model_para_dic = {
            'Model Name': '',
            'Hyper Parameter': {},

        }

        #  define model
        if USE_GPU:
            device = functional.try_gpu()
        else:
            device = 'cpu'
        dropout = 0.1
        nhd = 4
        nly = 3
        hid = 512
        model_name = 'PE_fixed_EC_transformer_DC_mlp_linear'
        net = model_define.PE_fixed_EC_transformer_DC_mlp_linear(d_model, tokenizer, nhd=nhd, nly=nly, dropout=dropout,
                                                                 hid=hid)
        # add graph
        temp_in = torch.rand(1, tokenizer.num_of_token, d_model)
        writer.add_graph(net, temp_in)
        net.to(device=device)

        # set optimizer
        lr = 0.001
        op = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(op, 1.0, gamma=0.95)

        # creat dataset
        train_dataset, batch_train = data_loader.creat_dataset(train_path, bsz=batch_size)
        val_dataset, batch_val = data_loader.creat_dataset(val_path, bsz=1)
        test_dataset, batch_test = data_loader.creat_dataset(test_path, bsz=1)

        """scalars_dic = {
            'train loss - epoch': train_epoch_loss,
            'val loss - epoch': val_epoch_loss,
        }
        writer.add_scalars('loss - epoch ', scalars_dic, epoch)

        # save model every 2 epoch
        if epoch % 2 == 0:
            save_name = f'{model_name}_{epoch}_{train_epoch_loss}_{val_epoch_loss}.pt'
            save_path = os.path.join(model_path, save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': op.state_dict(),
                'train_loss': train_epoch_loss,
                'val_loss': val_epoch_loss,
            }, save_path)"""

exit(0)
