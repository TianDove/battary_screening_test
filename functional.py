#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import numpy as np
import os
import math
import torch
import sklearn.metrics as metrics
from sklearn.preprocessing import FunctionTransformer


def sequenceDiff(in_sq):
    """计算差分序列"""
    sq_1 = in_sq[0:-1]
    sq_2 = in_sq[1:]
    sq_diff = sq_2 - sq_1
    return sq_diff


def round_precision(x, precision=0):
    """精确四舍五入"""
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


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
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
    def __init__(self, token_tuple=(32, True, 16)):
        """
        token_tup = (t_len, overlap, step)
        """
        self.t_len = token_tuple[0]
        self.overlap = token_tuple[1]
        self.step = token_tuple[2]
        self.detoken_len = None

    def calculate_expamle_detoken_len(self, example_path):
        """"""
        data_list = os.listdir(example_path)
        data_path = os.path.join(example_path, data_list[0])
        in_data = np.load(data_path)[0:-1]  # data specialize
        in_temp = in_data
        d_size = in_temp.shape
        r_mod = d_size[0] % self.t_len
        if not self.overlap:
            num_of_padding = 0
            if r_mod != 0:
                pad_num = in_temp[-1]
                num_of_padding = self.t_len - r_mod
            de_token_len = d_size[0] + num_of_padding
        else:
            num_of_step = math.ceil((d_size[0] - (self.t_len - self.step)) / self.step)
            de_token_len = (num_of_step - 1) * self.step + self.t_len
        self.detoken_len = de_token_len


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


def tokenize(in_data: np.array, t_len: int = 32, is_overlap: bool = True, step: int = 16):
    """
    in_data: array-like, shape (1, n_features)
    """
    assert len(in_data.shape) == 2 and in_data.shape[0] == 1
    in_temp = in_data
    d_size = in_temp.shape
    r_mod = d_size[1] % t_len
    if not is_overlap:
        if r_mod != 0:
            pad_num = in_temp[0, -1]
            num_of_padding = t_len - r_mod
            pad_arr = np.ones(num_of_padding) * pad_num
            pad_arr = np.expand_dims(pad_arr, axis=0)
            in_temp = np.concatenate((in_temp, pad_arr), axis=1)
        out_data = np.reshape(in_temp, (-1, t_len))
    else:
        num_of_step = math.ceil((d_size[1] - (t_len - step)) / step)
        detoken_len = (num_of_step - 1) * step + t_len
        if (detoken_len % d_size[1]) != 0:
            pad_num = in_temp[0, -1]
            num_of_padding = detoken_len - d_size[1]
            pad_arr = np.ones(num_of_padding) * pad_num
            pad_arr = np.expand_dims(pad_arr, axis=0)
            in_temp = np.concatenate((in_temp, pad_arr), axis=1)
        # overlap tokenize
        out_data = np.zeros((num_of_step, t_len))
        for stp in range(num_of_step):
            index = stp * step
            temp_token = in_temp[0, index:index + t_len]
            out_data[stp, :] = temp_token
    return out_data


def detokenize(in_data: np.array, t_len: int = 32, is_overlap: bool = True, step: int = 16):
    """
    in_data: array-like, shape (n_token, n_features)
    """
    assert len(in_data.shape) == 2
    org_size = in_data.shape
    if not is_overlap:
        out_data = in_data.reshape(1, -1)
    else:
        num_of_token = org_size[0]
        out_data = np.zeros((num_of_token - 1) * step + t_len)
        first_token = in_data[0, :]
        out_data[0:t_len] = first_token  # put first token into out sequence
        for i in range(1, num_of_token):
            curr_token = in_data[i, :]  # get token from second token
            curr_start_index = i * step
            curr_end_index = curr_start_index + t_len
            padded_curr_token = np.zeros((num_of_token - 1) * step + t_len)
            padded_curr_token[curr_start_index: curr_end_index] = curr_token
            out_data += padded_curr_token
            curr_mid_start_index = curr_start_index
            curr_mid_end_index = curr_start_index + step
            out_data[curr_mid_start_index: curr_mid_end_index] /= 2
    return np.expand_dims(out_data, axis=0)


def tokenizer(t_len: int = 32, is_overlap: bool = True, step: int = 16) -> FunctionTransformer:
    """"""
    para_dic = {'t_len': t_len, 'is_overlap': is_overlap, 'step': step}
    temp_tokenizer = FunctionTransformer(
        func=tokenize,
        inverse_func=detokenize,
        validate=True,
        accept_sparse=True,
        check_inverse=True,
        kw_args=para_dic,
        inv_kw_args=para_dic
    )
    return temp_tokenizer


if __name__ == '__main__':
    import data_loader
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    # dataset name
    dataset_name = 'train'
    dataset_path = '.\\data\\2600P-01_DataSet\\dataset\\' + dataset_name

    # tokenizer tup
    t_len = 32
    is_overlap = True
    step = 16
    token_tup = (t_len, is_overlap, step)
    batch_size = 64

    # tokenizer
    # tokenizer = Tokenizer(token_tup)

    # pre-process
    trans_tup = (
        Normalizer,
        StandardScaler,
        tokenizer
    )
    trans_para = (
        {'norm': 'l2', 'copy': True},
        {'copy': True, 'with_mean': True, 'with_std': True},
        {'t_len': 32, 'is_overlap': False, 'step': 16}
    )

    # creat dataset
    dataset, num_of_batch = data_loader.creat_dataset(dataset_path,
                                                      bsz=batch_size,
                                                      is_shuffle=True,
                                                      num_of_worker=0,
                                                      transform=trans_tup,
                                                      trans_para=trans_para)
    # batch loop
    for batch_index, raw_data in enumerate(dataset):
        temp_data = raw_data['data']
        temp_label = raw_data['label']
        print('')

