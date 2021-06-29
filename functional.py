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
import sklearn.metrics as metrics


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


def data_preprocessing(in_data=None):
    """"""
    if in_data is not None:
        d_shape = in_data.shape
        bsz = d_shape[0]
        d_numpy = in_data
        if torch.is_tensor(in_data):
            d_numpy = in_data.numpy()
        d_mean = d_numpy.mean(axis=1, keepdims=True)
        d_std = d_numpy.std(axis=1, keepdims=True)
        out_d = (d_numpy - d_mean) / d_std
        out_d = torch.from_numpy(out_d)
        return out_d, d_mean, d_std


def data_depreprocessing(in_data, d_mean, d_std):
    """"""
    if torch.is_tensor(in_data):
        d_numpy = in_data.numpy()
        out_d = (d_numpy * d_std) + d_mean
        out_d = torch.from_numpy(out_d)
        return out_d



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


def eval_wrapper(src, tgt):
    """"""

    explained_variance = metrics.explained_variance_score(tgt, src)
    max_error = metrics.max_error(tgt, src)
    mean_absolute_error = metrics.mean_absolute_error(tgt, src)
    mean_squared_error = metrics.mean_squared_error(tgt, src)
    mean_absolute_percentage_error = metrics.mean_absolute_percentage_error(tgt, src)
    median_absolute_error = metrics.median_absolute_error(tgt, src)
    r2_score = metrics.r2_score(tgt, src)

    eval_dic = {
        'explained_variance': explained_variance,  # 可解释方差
        'max_error': max_error,  # 误差的最大值
        'mean_absolute_error': mean_absolute_error,  # 误差绝对值的平均
        'mean_squared_error': mean_squared_error,  # 均方误差
        'mean_absolute_percentage_error': mean_absolute_percentage_error,  # 相对误差的平均
        'median_absolute_error': median_absolute_error,  # 误差绝对值的中位数
        'r2_score': r2_score   # R^2指标
    }
    return eval_dic


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
        in_temp = in_data
        d_size = in_temp.shape
        r_mod = d_size[0] % self.t_len
        if not self.overlap:
            if r_mod != 0:
                pad_num = in_temp[-1]
                num_of_padding = self.t_len - r_mod
                pad_arr = np.ones(num_of_padding) * pad_num
                in_temp = np.concatenate((in_temp, pad_arr))
            out_data = np.reshape(in_temp, (-1, self.t_len))
            num_of_token = out_data.shape[0]
        else:
            num_of_step = math.ceil((d_size[0] - (self.t_len - self.step)) / self.step)
            detoken_len = (num_of_step - 1) * self.step + self.t_len
            if (detoken_len % d_size[0]) != 0:
                pad_num = in_temp[-1]
                num_of_padding = detoken_len - d_size[0]
                pad_arr = np.ones(num_of_padding) * pad_num
                in_temp = np.concatenate((in_temp, pad_arr))
            # overlap tokenize
            out_data = np.zeros((num_of_step, self.t_len))
            for stp in range(num_of_step):
                index = stp * self.step
                temp_token = in_temp[index:index + self.t_len]
                out_data[stp, :] = temp_token
            num_of_token = out_data.shape[0]
        return out_data

    def detokenize(self, in_data):
        """"""
        org_size = in_data.shape
        if not self.overlap:
            out_data = in_data.view(1, -1)
        else:
            num_of_token = org_size[0]
            out_data = torch.zeros((num_of_token - 1) * self.step + self.t_len)
            first_token = in_data[0, :]
            out_data[0:self.t_len] = first_token  # put first token into out sequence
            for i in range(1, num_of_token):
                curr_token = in_data[i, :]  # get token from second token
                curr_start_index = i * self.step
                curr_end_index = curr_start_index + self.t_len
                padded_curr_token = torch.zeros((num_of_token - 1) * self.step + self.t_len)
                padded_curr_token[curr_start_index: curr_end_index] = curr_token
                out_data += padded_curr_token
                curr_mid_start_index = curr_start_index
                curr_mid_end_index = curr_start_index + self.step
                out_data[curr_mid_start_index: curr_mid_end_index] /= 2
        return out_data

    def token_wrapper(self, data, *args):
        """"""
        if args[0] == 'token':
            assert (len(data.shape) == 1) and (type(data) is np.ndarray)
            arr_token = self.tokenize(data)
        elif args[0] == 'detoken':
            # in_data is a tensor:(number of token, token length)
            assert torch.is_tensor(data) and (len(data.shape) == 2)
            arr_token = self.detokenize(data)
        else:
            raise Exception('Tokenize Mode Error.')
        # convert data
        re_data = arr_token
        return re_data



if __name__ == '__main__':
    """"""
    import data_loader

    # dataset name
    dataset_name = 'train'
    dataset_path = '.\\data\\2600P-01_DataSet\\dataset\\' + dataset_name

    # tokenizer tup
    t_len = 32
    is_overlap = True
    step = 16
    token_tup = (t_len, is_overlap, step)
    batch_size = 32

    # tokenizer
    tokenizer = Tokenizer(token_tup)
    # creat dataset
    dataset, num_of_batch = data_loader.creat_dataset(dataset_path, bsz=batch_size,
                                                      is_shuffle=True, num_of_worker=0)
    # batch loop
    file_list = os.listdir(dataset_path)
    for batch_index, raw_data_path in enumerate(file_list):
        raw_data = np.load(os.path.join(dataset_path, raw_data_path))
        temp_data = raw_data[0:-1]
        temp_label = raw_data[-1]
        data_tokenized = tokenizer.token_wrapper(temp_data, 'token')
        # de-tokenize
        de_token_in = torch.from_numpy(data_tokenized)
        data_detokenized = tokenizer.token_wrapper(de_token_in, 'detoken')
        print('')

