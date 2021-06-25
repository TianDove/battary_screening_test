#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import pandas as pd
import numpy as np
import os
import math
import pickle
import sys
import torch


def sequenceDiff(in_sq):
    """计算差分序列"""
    sq_1 = in_sq[0:-1]
    sq_2 = in_sq[1:]
    sq_diff = sq_2 - sq_1
    return sq_diff

def save_list_np(path, list_data):
    arr = np.array(list_data)
    np.save(path, arr)
    print(f"List File Saved:{path}.")


def load_list_np(path):
    arr = np.load(path)
    list = arr.tolist()
    print(f"List File Loaded:{path}.")
    return list


def xlsx_to_csv(src_path, out_path):
    xlsx_list = os.listdir(src_path)
    for i in iter(xlsx_list):
        file_path = os.path.join(src_path, i)
        xlsx = pd.read_excel(file_path)
        target_path = os.path.join(out_path, os.path.splitext(i)[0] + '.csv')
        xlsx.to_csv(target_path)


def round_precision(x, precision=0):
    """"""
    val = x * 10**precision
    int_part = math.modf(val)[1]
    fractional_part = math.modf(val)[0]
    out = 0
    if fractional_part >= 0.5:
        out = int_part + 1
    else:
        out = int_part
    out = out / 10**precision
    return out


def save_dic_as_pickle(target_path, data_dic):
    """"""
    if not os.path.exists(target_path):
        with open(target_path, 'wb') as f:
            pickle.dump(data_dic, f)
    else:
        print(f'Path Exist: {target_path}')
        sys.exit(0)


def load_dic_in_pickle(source_path):
    """"""
    if os.path.exists(source_path):
        with open(source_path, 'rb') as f:
            dic_data = pickle.load(f)
            return dic_data
    else:
        print(f'Path Not Exist: {source_path}')
        sys.exit(0)


class Tokenizer():
    """"""
    def __init__(self, token_tup):
        """
        token_tup = (t_len, overlap, step)
        """
        self.t_len = token_tup[0]
        self.overlap = token_tup[1]
        self.step = token_tup[2]
        self.detoken_shape = 0
        self.num_of_token = 0

    def calculate_expamle_detoken_len(self, example_path):
        """"""
        data_list = os.listdir(example_path)
        data_path = os.path.join(example_path, data_list[0])
        in_data = np.load(data_path)[0:-1]  # data specialize
        d_size = in_data.shape
        in_temp = in_data
        num_of_token = None
        detoken_shape = None
        num_of_padding = 0
        r_mod = d_size[0] % self.t_len
        if not self.overlap:
            if r_mod != 0:
                if r_mod < self.t_len // 2:
                    detoken_shape = d_size[0] - r_mod
                else:
                    num_of_padding = self.t_len - r_mod
                    detoken_shape = d_size[0] + num_of_padding
            num_of_token = detoken_shape // self.t_len
        else:
            if (d_size[0] % self.step) != (self.t_len - self.step):
                num_of_padding = self.t_len - self.step - r_mod
            detoken_shape = d_size[0] + num_of_padding
            num_of_token = (detoken_shape // self.step) - 1
        self.detoken_shape = detoken_shape
        self.num_of_token = num_of_token

    def tokenize(self, in_data):
        """"""
        d_size = in_data.shape
        in_temp = in_data
        r_mod = d_size[0] % self.t_len
        if not self.overlap:
            if r_mod != 0:
                if r_mod < self.t_len // 2:
                    in_temp = in_temp[0:-r_mod]
                else:
                    pad_num = in_temp[-1]
                    num_of_padding = self.t_len - r_mod
                    pad_tensor = torch.ones(num_of_padding) * pad_num
                    in_temp = torch.cat((in_temp, pad_tensor))
            out_data = in_temp.view(-1, self.t_len)
        else:
            if (d_size[0] % self.step) != (self.t_len - self.step):
                pad_num = in_temp[-1]
                num_of_padding = self.t_len - self.step - r_mod
                pad_tensor = torch.ones(num_of_padding) * pad_num
                in_temp = torch.cat((in_temp, pad_tensor))
            num_of_token = (in_temp.shape[0] // self.step) - 1
            out_data = torch.zeros((num_of_token, self.t_len))
            for stp in range(num_of_token):
                index = stp * self.step
                temp_token = in_temp[index:index + self.t_len]
                out_data[stp, :] = temp_token
        return out_data

    def detokenize(self, in_data):
        out_data = None
        org_size = in_data.shape
        if not self.overlap:
            out_data = in_data.view(1, -1)
        else:
            out_data = torch.zeros(1, (org_size[0] * self.step) + (self.t_len - self.step))
            out_data[0, 0: self.t_len].copy_(in_data[0, :])
            for i in range(org_size[0] - 1):
                padding_tensor = torch.zeros(self.step)
                t_1 = torch.cat((out_data[0, i * self.step: i * self.step + self.t_len], padding_tensor))
                t_2 = torch.cat((padding_tensor, in_data[i + 1, :]))
                temp = t_1 + t_2
                mid = temp[self.step: self.step + (self.t_len - self.step)] / 2
                temp[self.step: self.step + (self.t_len - self.step)] = mid
                out_data[0, i * self.step: i * self.step + self.t_len + self.step] = temp
        return out_data

    def token_wrapper(self, data_batch, mode='token'):
        """"""
        data_shape = data_batch.shape
        data_token_batch = []
        for i in range(data_shape[0]):
            if mode == 'token':
                arr_token = self.tokenize(data_batch[i])
                arr_token = arr_token.unsqueeze(0)
            elif mode == 'detoken':
                arr_token = self.detokenize(data_batch[i])
            else:
                raise Exception('Tokenize Mode Error.')
            data_token_batch.append(arr_token)
        # convert data
        re_data = torch.cat(data_token_batch, dim=0)
        return re_data

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
    """"""
    """is_over_lap = True
    t_len = 32
    example = np.load('.\\data\\2600P-01_DataSet\\dataset\\val\\210301-1_C000000397_4.npy')
    data = example[:-1]
    label = example[-1]
    sample = tokenize(data, t_len=t_len, overlap=is_over_lap)
    out = detokenize(sample, t_len=t_len, overlap=is_over_lap)"""
    import matplotlib
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    file_path = '.\\data\\2600P-01_DataSet\\pickle'
    file_list = os.listdir(file_path)
    fig, ax = plt.subplots()
    with tqdm(total=len(file_list)) as bar:
        bar.set_description('Loading Data...')
        for i, file in enumerate(iter(file_list)):
            bar.update()
            file_name = os.path.join(file_path, file)
            data = load_dic_in_pickle(file_name)
            temp_form_ocv_1 = data['Static']['Form-OCV #1'].astype('float32', 1).item()
            if 3300 < temp_form_ocv_1 and temp_form_ocv_1 < 3900:
                ax.scatter(i, temp_form_ocv_1)
                plt.pause(0.001)
    plt.show()

