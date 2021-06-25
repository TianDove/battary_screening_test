#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import os
import sys

import pandas as pd
import torch.utils.data as t_u_data
import random
import math
from tqdm import tqdm
import copy
import numpy as np

#
import file_operation
import functional


def creat_dataset(data_path, bsz=32, is_shuffle=True, num_of_worker=None):
    """"""
    assert os.path.exists(data_path)
    data_set = CellDataSet(data_path)
    batch_data_set = t_u_data.DataLoader(data_set, batch_size=bsz, shuffle=is_shuffle)
    num_of_batch = math.floor(len(os.listdir(data_path)) / bsz)
    return batch_data_set, num_of_batch


class CellDataSet(t_u_data.Dataset):
    """Pytorch DataSet"""
    def __init__(self, path):
        self.file_path = path
        self.list_files = os.listdir(self.file_path)
        self.num_of_samples = len(self.list_files)

        # join path
        self.file_path_list = []
        for i in iter(self.list_files):
            temp_file_path = os.path.join(self.file_path, i)
            self.file_path_list.append(temp_file_path)


    def __getitem__(self, index):
        data = np.load(self.file_path_list[index])
        return data


    def __len__(self):
        return len(self.file_path_list)


class DataLoader():
    """"""
    def __init__(self, data_path, divided_path, param_mod_dic):
        self.dataPath = data_path
        self.dividedPath = divided_path
        self.npy_path = os.path.split(data_path)[0] + '\\npy'
        self.paramModDic = param_mod_dic
        self.used_key = []

    def data_convert(self, cells_dic):
        """"""
        cell_list = list(cells_dic.keys())
        dic_extracted_converted = {}
        with tqdm(total=len(cell_list)) as bar:
            bar.set_description('Data Converting...')
            for cell in iter(cell_list):
                bar.update()
                temp_data_dic = cells_dic[cell]
                for key in iter(temp_data_dic.keys()):
                    if key == 'Static':
                        temp_data_dic[key] = temp_data_dic[key].astype('float32', 1)
                    else:
                        temp_data_dic[key] = temp_data_dic[key].iloc[1:, :].astype('float32', 1)
                dic_extracted_converted.update({f'{cell}': temp_data_dic})
        return dic_extracted_converted

    def data_loder(self):
        """"""
        # if self.check_dir():
        cells_data_list = os.listdir(self.dataPath)
        cells_dic = self.data_Load(cells_data_list)
            # if cells_dic != {}:
        cells_dic_selected = self.cell_data_selection(cells_dic)
        # specified for charge #1 and Form-OCV #1
        cells_dic_converted = self.data_convert(cells_dic_selected)
        self.ch1_data_clean_and_interpolation(cells_dic_converted)
        return 0

    def ch1_data_clean_and_interpolation(self, data_dic):
        """"""
        para_name = 'Charge #1'
        for cell in iter(data_dic):
            temp_df = data_dic[cell][para_name]
            field_list = list(temp_df.columns)
            temp_time = data_dic[cell][para_name]['time']
            field_list.remove('time')
            temp_res = data_dic[cell][para_name][field_list]
            for j, field in enumerate(iter(field_list)):
                temp_str = cell + '_' + field
                field_list[j] = temp_str
            df_index_time = pd.DataFrame(temp_res.values, index=temp_time.values, columns=field_list, )
            data_dic[cell][para_name] = df_index_time.copy()
        #
        
        dic_key = iter(data_dic)
        first_df = data_dic[next(dic_key)][para_name]
        with tqdm(total=len(list(data_dic.keys()))) as bar:
            bar.set_description('Charge #1 Data Concatenating....')
            for key in dic_key:
                bar.update()
                first_df = pd.concat([first_df, data_dic[key][para_name]], axis=1)
        #
        row_list = list(first_df.index)
        del_row_list = []
        for row in iter(row_list):
            index = row
            res = index % 0.5
            if res != 0:
                del_row_list.append(index)
        #
        align_df = first_df.drop(del_row_list).copy()
        print('Interpolation Processing....', end='')
        align_df = align_df.interpolate(method='quadratic', axis=0)
        print('\r' + 'Interpolation Complete.')
        align_df_values = align_df.values
        mean = np.mean(align_df_values, axis=1, keepdims=True)
        std = np.std(align_df_values, axis=1, keepdims=True, ddof=1)
        temp_sq = align_df_values
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std
        a1 = (lower_bound < temp_sq)
        a2 = (temp_sq < upper_bound)
        a = np.logical_and(a1, a2)
        a_sum = a.sum(axis=0, keepdims=True)
        a_index = a_sum < (a.shape[0]/2)  # True for del col
        del_index = np.where(a_index == True)
        align_df_col_list = list(align_df.columns)
        index_to_col_list = []
        for index in iter(del_index[1]):
            temp_col = align_df_col_list[index]
            index_to_col_list.append(temp_col)
        align_df = align_df.drop(index_to_col_list, axis=1)
        col_list = list(align_df.columns)
        # res_size = [align_df.shape[0] + 1, align_df.shape[1]]
        # res_arr = np.zeros(res_size)
        with tqdm(total=len(col_list)) as w_bar:
            w_bar.set_description('Charge #1 Data Writing....')
            for t, col in enumerate(iter(col_list)):
                w_bar.update()
                str_split = col.split('_')
                cell_name = str_split[0] + '_' +str_split[1] + '_' +str_split[2]
                static_data = data_dic[cell_name]['Static'].values.squeeze(axis=1)
                dynamic_data = align_df[col].values
                sample_with_label = np.concatenate([dynamic_data, static_data])
                file_name = cell_name + '.npy'
                save_path = os.path.join(self.npy_path, file_name)
                np.save(save_path, sample_with_label)
        return 0


    def check_dir(self):
        """"""
        check_flag = False
        print(f'Input Data Path:{self.dataPath}')
        print(f'Output Data Path:{self.dividedPath}')
        key_list = self.paramModDic.keys()
        for i in iter(key_list):
            temp_list = self.paramModDic[i]
            print(f'Selected Param--{i}--{temp_list}')
        if os.path.exists(self.dataPath):
            # get train and test data path
            train_path = self.dividedPath + '\\train'
            test_path = self.dividedPath + '\\test'
            val_path = self.dividedPath + '\\val'
            if os.path.exists(self.dividedPath):
                file_operation.Delete_File_Dir(self.dividedPath)
                print(f'Dir Removed:{self.dividedPath}')
            os.mkdir(self.dividedPath)
            print(f'Dir Created:{self.dividedPath}')
            os.mkdir(train_path)
            print(f'Dir Created:{train_path}')
            os.mkdir(test_path)
            print(f'Dir Created:{test_path}')
            os.mkdir(val_path)
            print(f'Dir Created:{val_path}')
            check_flag = True
        else:
            print(f'Input Data Dir No Found:{self.dataPath}')
            sys.exit(0)
        return check_flag

    def data_divide(self, data_set_path, ratio=0.1):
        """"""
        file_list = os.listdir(data_set_path)
        selected_samples = file_list
        random.shuffle(selected_samples)
        shuffled_sample_list = selected_samples
        num_of_samples = len(selected_samples)
        random.shuffle(shuffled_sample_list)
        val_num = math.floor(num_of_samples * ratio)
        test_num = math.floor(num_of_samples * ratio)
        train_num = num_of_samples - val_num - test_num
        if (val_num + test_num + train_num) == num_of_samples:
            val_list = shuffled_sample_list[0: val_num]
            test_list = shuffled_sample_list[val_num: val_num + test_num]
            train_list = shuffled_sample_list[val_num + test_num: val_num + test_num + train_num]
            #
            train_fold = os.path.join(self.dividedPath, 'train')
            val_fold = os.path.join(self.dividedPath, 'val')
            test_fold = os.path.join(self.dividedPath, 'test')
            file_operation.move_file_to_fold(self.npy_path, train_fold, train_list)
            file_operation.move_file_to_fold(self.npy_path, val_fold, val_list)
            file_operation.move_file_to_fold(self.npy_path, test_fold, test_list)
            print(f'Data Dividing Completed.')

        else:
            print(f'Sample Num Validation Error.')
            sys.exit(0)
            
    def data_Load(self, path_list, mode='train'):
        """"""
        data = {}
        if path_list != []:
            print(f'{mode} Data Loading Start.')
            with tqdm(total=len(path_list)) as data_read_bar:
                data_read_bar.set_description('Cell Data Reading...')
                for i in iter(path_list):
                    cell_path = os.path.join(self.dataPath, i)
                    cell_name = os.path.splitext(i)[0]
                    cell_dic = functional.load_dic_in_pickle(cell_path)
                    data.update({f'{cell_name}': cell_dic})
                    data_read_bar.update()
        return data

    def cell_data_selection(self, data_dic):
        """"""
        cells_list = list(data_dic.keys())
        cells_dic = {}
        for use_key in iter(self.paramModDic.keys()):
            if self.paramModDic[use_key]:
                self.used_key.append(use_key)
        with tqdm(total=len(cells_list)) as bar:
            bar.set_description('Cell Data Selecting....')
            list_extracted = copy.deepcopy(cells_list)
            for cell_name in iter(cells_list):
                cell_dic = {}
                bar.update()
                selected_flag = True
                # get the key of parameter of the current cell
                cell_key = list(data_dic[cell_name].keys())
                # whether cell have used parameter
                for key in iter(self.used_key):
                    if key not in cell_key:
                        selected_flag = False
                        break
                if selected_flag:
                    for used in iter(self.used_key):
                        para_list = self.paramModDic[used]
                        para = data_dic[cell_name][used][[x for x in iter(para_list)]]
                        cell_dic.update({f'{used}': para})
                    cells_dic.update({f'{cell_name}': cell_dic})
        return cells_dic


if __name__ == '__main__':

    """param_mode_dic = {
        'Static': ['Form-OCV #1', ],
        'Charge #1': ['time', 'voltage'],
        'Charge #2': [],
        'Charge #3': [],
        'Discharge': [],
        'Charge #4': []
    }
    cells_data_path = '.\\data\\2600P-01_DataSet\\pickle'  # Load pickle data
    cells_divided_data_path = '.\\data\\2600P-01_DataSet\\dataset'  # path store divided dataset
    m_dataLoader = DataLoader(cells_data_path, cells_divided_data_path, param_mode_dic)
    m_dataLoader.data_loder()
    m_dataLoader.data_divide(m_dataLoader.npy_path)
    batch_size = 32
    train_path = '.\\data\\2600P-01_DataSet\\dataset\\train'
    cell_data_set = CellDataSet(train_path)
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
    

   train_path = '.\\data\\2600P-01_DataSet\\dataset\\train'
    tokenize_para = (32, True, 16)
    data_set, num_of_batch = creat_dataset(train_path)
    m_tokenizer = functional.Tokenizer(tokenize_para)
    detoken_shape, num_of_token = m_tokenizer.calculate_expamle_detoken_len(train_path)
    for i, data in enumerate(data_set):
        temp_data = data[:, 0:-1]
        temp_label = data[:, -1]
        token = m_tokenizer.token_wrapper(temp_data, mode='token', para_tup=tokenize_para)
        detoken = m_tokenizer.token_wrapper(token, mode='detoken', para_tup=tokenize_para)"""

    sys.exit(0)
