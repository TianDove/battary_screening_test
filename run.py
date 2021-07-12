#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import time
import os
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

# import self defined
import file_operation

# constant
BATCH_LOG_INTERVAL = 10

# set net loss function
CRITERION = nn.MSELoss()


class Run():
    """

    """
    def __init__(self,
                model_name: str,
                model: nn.Module,
                optimizer: torch.optim,
                loss: nn.Module,
                dataset_dic: dict,
                scheduler: torch.optim.lr_scheduler = None,
                device: torch.device = 'cpu',
                writer: SummaryWriter = None) -> None:
        """
        dataset_dic:dictionary contain train, validation and test data set,
                    like {'train_data_set':..., 'num_of_train_batch':...,
                          'val_data_set':...,'num_of_val_batch'...,
                          'test_data_set':..., 'num_of_test_batch':...}
        """
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.dataset_dic = dataset_dic
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.loss = loss

        # loss variable
        self.batch_train_loss = 0
        self.batch_log_interval_train_loss = 0
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.test_loss = 0
        self.epoch_val_loss_list = []
        self.test_loss_list = []

        # init time dictionary
        self.record_time_dic = {
            'epoch_start_time': 0.0,
            'train_start_time': 0.0,
            'train_end_time': 0.0,
            'train_batch_start_time': 0.0,
            'train_batch_end_time': 0.0,
            'val_start_time': 0.0,
            'epoch_end_time': 0.0,
            'test_start_time': 0.0,
            'test_end_time': 0.0
        }
        # current loop state
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_stage = None

    def train_model(self, num_epochs: int = 3) -> None:
        """

        """
        # epoch loop
        for epoch in range(num_epochs):
            self.curr_epoch = epoch  # log current epoch index
            self.record_time_dic['epoch_start_time'] = time.time()  # log epoch start time
            # setup training
            self.model.train()
            self.epoch_train_loss = 0


            # batch loop
            self.record_time_dic['train_start_time'] = time.time()  # log training start time
            for batch_index, train_data in enumerate(self.dataset_dic['train_data_set']):
                self.curr_stage = 'train'  # log current stage
                self.curr_batch = batch_index  # log current batch index
                self.record_time_dic['train_batch_start_time'] = time.time()  # log training batch start time
                self.batch_train_loss = 0
                # process
                input_data = train_data['data'].to(self.device)
                input_label = train_data['label'].to(self.device)
                self.optimizer.zero_grad()
                out = self.model(input_data)
                if len(out.shape) > 1:
                    out = out.squeeze(1)
                scalar_loss = self.loss(out, input_label)
                self.batch_train_loss = scalar_loss.detach().item()
                self.epoch_train_loss += scalar_loss.detach().item()
                out.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.record_time_dic['train_batch_end_time'] = time.time()  # log training batch end time
            self.record_time_dic['train_end_time'] = time.time()  # log training end time
            # calculate average train loss of epoch
            self.epoch_train_loss /= self.dataset_dic['num_of_train_batch']

            # setup validation
            self.model.eval()
            self.epoch_val_loss = 0
            self.epoch_val_loss_list = []
            self.record_time_dic['val_start_time'] = time.time()  # log valid start time
            with torch.no_grad():
                for val_index, val_data in enumerate(self.dataset_dic['val_data_set']):
                    self.curr_stage = 'val'
                    input_data = val_data['data'].to(self.device)
                    input_label = val_data['label'].to(self.device)
                    out = self.model(input_data)
                    if len(out.shape) > 1:
                        out = out.squeeze(1)
                    scalar_loss = self.loss(out, input_label)
                    self.epoch_val_loss += scalar_loss.item()
                    self.epoch_val_loss_list.append(scalar_loss.item())
            self.record_time_dic['epoch_end_time'] = time.time()  # log epoch end time
            # calculate average valid loss of epoch
            self.epoch_val_loss /= self.dataset_dic['num_of_val_batch']

    def test_model(self) -> None:
        """

        """
        # setup validation
        self.model.eval()
        self.test_loss = 0
        self.test_loss_list = []
        self.record_time_dic['test_start_time'] = time.time()  # log test start time
        with torch.no_grad():
            for test_index, test_data in enumerate(self.dataset_dic['test_data_set']):
                self.curr_stage = 'test'
                input_data = test_data['data'].to(self.device)
                input_label = test_data['label'].to(self.device)
                out = self.model(input_data)
                if len(out.shape) > 1:
                    out = out.squeeze(1)
                scalar_loss = self.loss(out, input_label)
                self.test_loss += scalar_loss.item()
                self.test_loss_list.append(scalar_loss.item())
        self.record_time_dic['test_end_time'] = time.time()  # log test end time
        # calculate average test loss of epoch
        self.test_loss /= self.dataset_dic['num_of_test_batch']

    def log(self, log_file_path, mode: str = 'train') -> None:
        """"""
        str_data = ''
        separator = ''
        if mode == 'train':
            # log batch train
            separator = '-' * 89
            str_data = '| Epoch {:3d} | {:5d}/{:5d} batches | lr {:10.9f} | ms/batch {:5.2f} | Loss {:10.9f} |'.format(
                    self.curr_epoch,
                    self.curr_batch, self.dataset_dic['num_of_train_batch'] - 1,
                    self.optimizer.param_groups[0]['lr'],
                    self.record_time_dic['train_batch_end_time'] - self.record_time_dic['train_batch_start_time'],
                    self.batch_train_loss)

        elif mode == 'val':
            # log val
            separator = '-' * 89
            str_data = '| End of epoch {:3d} | Val Total Time: {:5.2f}s | Avg Valid Loss {:10.9f} | '.format(
                        self.curr_epoch,
                        self.record_time_dic['epoch_end_time'] - self.record_time_dic['val_start_time'],
                        self.epoch_val_loss)

        elif mode == 'test':
            # log test
            separator = '#' * 89
            str_data = '| Test Stage | Test Total Time: {:5.2f}s | Avg Test Loss {:10.9f} | '.format(
                self.record_time_dic['test_end_time'] - self.record_time_dic['test_start_time'],
                self.test_loss)
        else:
            raise ValueError('Input Log Mode Error.')

        # write to file
        if str_data != '':
            file_operation.write_txt(log_file_path, separator)
            file_operation.write_txt(log_file_path, str_data)
            file_operation.write_txt(log_file_path, separator)
        else:
            raise ValueError('String Data Container Empty Error.')

    def save_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        save_name = f'{self.model_name}_{self.curr_epoch}.pth'
        save_path = os.path.join(path, save_name)
        torch.save({
                'model_name': self.model_name,
                'epoch': self.curr_epoch,
                'model': self.model,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'train_loss': self.epoch_train_loss,
                'val_loss': self.epoch_val_loss,
            }, save_path)

    def load_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        temp_dict = torch.load(path)
        self.model_name = temp_dict['model_name']
        self.curr_epoch = temp_dict['epoch']
        self.model = temp_dict['model']
        self.optimizer = temp_dict['optimizer']
        self.scheduler = temp_dict['scheduler']
        self.epoch_train_loss = temp_dict['train_loss']
        self.epoch_val_loss = temp_dict['val_loss']


if __name__ == '__main__':
    pass