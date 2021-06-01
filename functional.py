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

def sequenceDiff(in_sq):
    """计算差分序列"""
    sq = -1 * in_sq[0:-1]
    sq_index = sq.keys()
    sq_shift = pd.Series(in_sq.iloc[1:].to_list(), index=sq_index)
    sq_diff = sq_shift.add(sq)
    return sq_diff

def scalarDiff(x):
    pass

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


if __name__ == '__main__':
    t = round_precision(3.14159, 4)
