#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import os
import sys
import math
import time
import copy
import shutil
import datetime
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# import self define module
import file_operation
import functional

# constant
NUM_CELL_IN_TRAY = 256


class FileOrganizer():
    """"""
    def __init__(self, in_path: str, out_path: str, is_multi_worker: bool = False):
        """"""
        # save input
        self.in_path = in_path
        self.out_path = out_path
        self.is_multi_worker = is_multi_worker

        # class path
        self.dynamic_data_path = None
        self.static_data_path = None

        # class state
        self.is_write = None
        self.curr_batch = None
        self.curr_tray = None
        self.curr_cell = None
        self.curr_para = None

        # dynamic parameter index
        self.num_of_cell_in_tray = NUM_CELL_IN_TRAY
        self.voltage_index = [x for x in range(1, self.num_of_cell_in_tray + 1)]
        self.current_index = [x for x in range(self.num_of_cell_in_tray + 1, self.num_of_cell_in_tray * 2 + 1)]
        self.capacity_index = [x for x in range(self.num_of_cell_in_tray * 2 + 1, self.num_of_cell_in_tray * 3 + 1)]

    def make_file_dir(self):
        """"""
        #################################
        # Raw Data Fold Structure
        # 2600P-01
        # |
        # | -- dynamic
        # |    |
        # |    | -- 210301-1
        # |    |    |
        # |    |    | -- 0397
        # |    |    |    |
        # |    |    |    | -- 0397-1.csv
        # |    |    |    | -- 0397-2.csv
        # |    |    |    | -- ...
        # |    |    | -- 0640
        # |    |    | -- ...
        # |    |
        # |    | -- 210301-2
        # |    | -- ...
        # |
        # | -- static
        #      |
        #      | -- 210301-1.csv
        #      | -- 210301-2.csv
        #      | -- ...
        #################################
        # Output Data Fold Structure
        # 2600P-01_DataSet
        # |
        # | -- organized_data
        # |    |
        # |    | -- npy
        # |    | -- pickle
        # |    | -- xlsx
        # |
        # | -- data_set
        #      |
        #      | -- all
        #      | -- train
        #      | -- val
        #      | -- test
        #################################
        assert os.path.exists(self.in_path)
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
            og_path = os.path.join(self.out_path, 'organized_data')
            os.mkdir(og_path)
            og_sub_list = ['npy', 'pickle', 'xlsx']
            for fold in iter(og_sub_list):
                temp_path = os.path.join(og_path, fold)
                os.mkdir(temp_path)

            ds_path = os.path.join(self.out_path, 'data_set')
            os.mkdir(ds_path)
            ds_sub_list = ['all', 'train', 'val', 'test']
            for fold in iter(ds_sub_list):
                temp_path = os.path.join(ds_path, fold)
                os.mkdir(temp_path)

    @staticmethod
    def tray_name_adjust(tray: str) -> str:
        """"""
        tray = tray.strip('-')
        non_zero_str = ''
        if len(tray) > 0:
            len_of_tray_id = len('C000009429')
            for i, num in enumerate(iter(tray)):
                if num != '0':
                    non_zero_str = tray[i:]
                    break
            pad_num = len_of_tray_id - len(non_zero_str) - 1
            new_tray = 'C' + '0' * pad_num + non_zero_str
            return new_tray
        else:
            print('Current Tray ID Empty.')
            sys.exit(0)

    @ staticmethod
    def parameter_name_adjust(para: str) -> str:
        """"""
        para_name_list = os.path.splitext(para)
        while para_name_list[-1] != '':
            para_name_list = os.path.splitext(para_name_list[0])
        para_name = para_name_list[0]
        new_para_name = ''
        if len(para_name) > 0:
            if '-' in para_name:
                if para_name[-1] == '1':
                    new_para_name = 'Charge #1'
                elif para_name[-1] == '2':
                    new_para_name = 'Charge #2'
                elif para_name[-1] == '3':
                    new_para_name = 'Charge #3'
                elif para_name[-1] == '4':
                    new_para_name = 'Discharge'
                elif para_name[-1] == '5':
                    new_para_name = 'Charge #4'
                else:
                    print('Current Parameter Index Outer Of Range.')
                    sys.exit(0)
            else:
                print('Current Parameter Name Error.')
                sys.exit(0)
            return new_para_name

        else:
            print('Current Parameter Name Empty.')
            sys.exit(0)

    @staticmethod
    def cell_no_to_tray_cell_no(cell_no: float) -> str:
        """"""
        cell_no = int(cell_no)
        alp = ('A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P')
        res = (cell_no - 1) % 16
        times = int((cell_no - 1) / 16)
        tray_cell_no = alp[times] + str(res + 1)
        return tray_cell_no

    def read_static_data(self, batch: str) -> pd.DataFrame:
        """"""
        current_batch_static_data_path = os.path.join(self.static_data_path, batch + '.csv')
        assert os.path.getsize(current_batch_static_data_path)
        static_data = pd.read_csv(open(current_batch_static_data_path, 'rb'), low_memory=False)
        return static_data

    def read_dynamic_para_tray(self, para_path: str, para_name: str) -> pd.DataFrame:
        """"""
        current_para_path = os.path.join(para_path, para_name)
        assert os.path.getsize(current_para_path)
        current_para_data = pd.read_csv(current_para_path)
        return current_para_data

    def get_batch_to_process(self) -> list:
        # dynamic data path and batch list
        self.dynamic_data_path = os.path.join(self.in_path, 'dynamic')
        dynamic_batch_list = os.listdir(self.dynamic_data_path)

        # static data path and batch list
        self.static_data_path = os.path.join(self.in_path, 'static')
        static_batch_list = os.listdir(self.static_data_path)

        # get batch
        temp_batch_list = []
        for batch in iter(dynamic_batch_list):
            if (batch + '.csv') in static_batch_list:
                temp_batch_list.append(batch)
        return temp_batch_list

    def get_tray_to_process(self, is_balance=False) -> list:
        """
        static_df: pd.DataFrame, batch: str, tray: str, tray_path: str
        """
        batch_list = self.get_batch_to_process()
        temp_tray_list = []

        if is_balance:
            num_tray_per_batch = []
            tray_index_per_batch = {}
            for batch in iter(batch_list):
                current_batch_tray_path = os.path.join(self.dynamic_data_path, batch)
                tray_list = os.listdir(current_batch_tray_path)
                num_tray_per_batch.append(len(tray_list))
                tray_index_per_batch.update({f'{batch}': tray_list})

        for batch in iter(batch_list):
            curr_batch = batch
            current_batch_tray_path = os.path.join(self.dynamic_data_path, batch)
            tray_list = os.listdir(current_batch_tray_path)
            current_batch_static_data = self.read_static_data(batch)
            for curr_tray in iter(tray_list):
                curr_tray_path = os.path.join(current_batch_tray_path, curr_tray)
                temp_tray_worker_dic = {
                    'static_df': current_batch_static_data,
                    'batch': curr_batch,
                    'tray': curr_tray,
                    'tray_path': curr_tray_path
                }
                temp_tray_list.append(temp_tray_worker_dic)
        return temp_tray_list

    def dynamic_para_extract_cell(self, cell_no: int, para_data: pd.DataFrame) -> pd.DataFrame:
        """"""
        temp_dic = {}
        # time
        time = para_data.iloc[:, 0]
        temp_dic.update({'time': time})
        # voltage
        voltage = para_data.iloc[:, cell_no]
        temp_dic.update({'voltage': voltage})
        # current
        current = para_data.iloc[:, cell_no + self.num_of_cell_in_tray]
        temp_dic.update({'current': current})
        # capacity
        capacity = para_data.iloc[:, cell_no + (2 * self.num_of_cell_in_tray)]
        temp_dic.update({'capacity': capacity})
        # build dataframe
        temp_df = pd.DataFrame(temp_dic)
        return temp_df

    def data_verification(self, static_df: pd.DataFrame, para_df: pd.DataFrame, ver_type: str) -> bool:
        """"""
        # Charge  #1, Charge #2, Charge #3, Discharge, Charge #4
        # |            |          |          |          |
        # |-voltage    |-capacity |-capacity |-capacity |-capacity
        ver_flag = False
        ver_num = 0
        tgt = 0
        if ver_type == 'Charge #1':
            ver = para_df['voltage'].iloc[-1]
            tgt = int(static_df[ver_type + '.2'].iloc[-1])
            ver_rounded = functional.round_precision(float(ver) * 1000.0)
            ver_num = int(ver_rounded)
        elif ver_type != 'Charge #1':
            ver = para_df['capacity'].iloc[-1]
            tgt = int(static_df[ver_type].iloc[-1])
            ver_rounded = functional.round_precision(float(ver))
            ver_num = int(ver_rounded)
        else:
            print('Verification Stage: Parameter No Found.')
            sys.exit(0)
        if abs(ver_num - tgt) < 3:
            ver_flag = True
        return ver_flag

    def file_organize_work_in_tray(self, is_write: bool = True):
        self.is_write = is_write
        self.make_file_dir()
        batch_tray_list = self.get_tray_to_process(is_balance=True)
        func_iter = iter(batch_tray_list)

        # output
        in_comp_cells = []
        err_para_cells = []

        # start processing
        start_time = time.time()
        if self.is_multi_worker:
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                res = pool.map(self.tray_worker, func_iter)
            end_time = time.time()
            self.log_file_organie(start_time, end_time, res)
        else:
            print('Single Processing, Pleas Using :self.file_organize_work_in_batch(), '
                  'with self.is_multi_worker = False.')


    def file_organize_work_in_batch(self, is_write: bool = True):
        """"""
        self.is_write = is_write
        self.make_file_dir()
        batch_list = self.get_batch_to_process()

        start_time = time.time()
        # use multiprocess or not
        if self.is_multi_worker:
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                res = pool.map(self.batch_worker, batch_list)
            end_time = time.time()
        else:
            res = []
            for batch in batch_list:
                temp_res = self.batch_worker(batch)
                res.append(temp_res)
            end_time = time.time()
        # log
        self.log_file_organie(start_time, end_time, res)
        print('Logging Finished, All Done.')

    def log_file_organie(self, start, end, res: list):
        """"""
        # logging
        now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(self.out_path, now_time + '_' + 'FileOrganizer_Logging.txt')
        consumed_time = end - start

        seperator = '-' * 89
        file_operation.write_txt(log_file_path, seperator)
        text_data = '| Finish Date: {} | Write Flag: {} | Multiprocess Flag: {} | Time Consume: {} s |'.format(
            now_time, self.is_write, self.is_multi_worker, consumed_time)
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Incomplete Cells List ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in res:
            if (i != {}) and (i['Incomplete Cells List'] != []):
                text_data = i['Incomplete Cells List']
                file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Verification Failed Cells ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in res:
            if (i != {}) and (i['Verification Failed Cells'] != []):
                text_data = i['Verification Failed Cells']
                file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

    def batch_worker(self, batch: str) -> (list, list):
        """"""
        if self.is_multi_worker:
            print('# {} Processing Start, PID: {}'.format(batch, os.getpid()))
        err_para_cells_list = []
        in_compelet_cells_list = []
        batch_cells_dic = {}
        curr_batch = batch
        # read current batch static file
        current_batch_static_data = self.read_static_data(batch)
        # current batch tray list
        current_batch_tray_path = os.path.join(self.dynamic_data_path, batch)
        tray_list = os.listdir(current_batch_tray_path)

        # progress tray_bar parameter setup
        tray_total = len(tray_list)
        text = "#{}-Processing..., PID: {}".format(curr_batch, os.getpid())
        with tqdm(total=tray_total, desc=text) as tray_bar:
            for tray in iter(tray_list):
                tray_bar.update()
                tray_name = self.tray_name_adjust(tray)
                curr_tray = tray_name
                current_tray_static_data = current_batch_static_data[current_batch_static_data['Tray ID']
                                                                     == tray_name]
                assert current_tray_static_data.shape[0] == self.num_of_cell_in_tray
                current_batch_tray_params_path = os.path.join(current_batch_tray_path, tray)
                params_list = os.listdir(current_batch_tray_params_path)
                paras_tray_dic = {}
                for para_tray in iter(params_list):
                    # read current batch-tray-parameter
                    para_name = self.parameter_name_adjust(para_tray)
                    current_para_tray_data = self.read_dynamic_para_tray(current_batch_tray_params_path, para_tray)
                    current_para_tray_data.iloc[0, 0] = curr_batch + '-' + para_tray
                    paras_tray_dic.update({f'{para_name}': current_para_tray_data})
                cell_no = [k for k in range(1, self.num_of_cell_in_tray + 1)]

                # progress cell_bar parameter setup
                #cell_total = len(cell_no)
                #text = "#{}-Processing...".format(curr_tray)
                #with tqdm(total=cell_total, desc=text) as cell_bar:
                for cell in iter(cell_no):
                    #cell_bar.update()
                    paras_dic = {}
                    err_para_list = []
                    curr_cell = curr_batch + '_' + curr_tray + '_' + str(cell)
                    # get cell's static data
                    static_df = current_tray_static_data.iloc[cell - 1, 1:].to_frame().astype('str', 1)
                    static_df_trans = pd.DataFrame(static_df.values.T,
                                                   index=static_df.columns,
                                                   columns=static_df.index)
                    static_df_cp = static_df_trans.copy()
                    paras_dic.update({'Static': static_df_cp})
                    if len(paras_tray_dic.keys()) < 5:
                        in_compelet_cells_list.append(curr_cell)
                    for para in iter(paras_tray_dic.keys()):
                        # verify, such as:   Charge #1          Other
                        #                  End Voltage(mV)   Capacity(mAH)
                        curr_para = para
                        # para extract mode:time, voltage, current, capacity
                        para_df = self.dynamic_para_extract_cell(cell, paras_tray_dic[para])
                        para_ver_flag = self.data_verification(static_df_trans, para_df, para)
                        if (not para_df.empty) and para_ver_flag:
                            paras_dic.update({f'{para}': para_df.copy()})
                            # update paras_dic to cell_dic
                        elif not para_ver_flag:
                            err_para_list.append(curr_cell + '_' + curr_para)
                    if err_para_list:
                        err_para_cells_list.append(curr_cell)
                    # update cells_dic
                    # batch_cells_dic.update({f'{curr_cell}': paras_dic.copy()})
                    if self.is_write:
                        write_xlsx_path = os.path.join(self.out_path + '\\organized_data\\xlsx', curr_cell + '.xlsx')
                        if not os.path.exists(write_xlsx_path):
                            with pd.ExcelWriter(write_xlsx_path) as writer:
                                # print(f'Write Cell File: {self.curr_cell}.', end='')
                                for name in iter(paras_dic.keys()):
                                    paras_dic[name].to_excel(writer, sheet_name=name)
                        # write cell pickle file
                        write_pickle_path = os.path.join(self.out_path +
                                                         '\\organized_data\\pickle', curr_cell + '.pickle')
                        if not os.path.exists(write_pickle_path):
                            file_operation.save_dic_as_pickle(write_pickle_path, paras_dic)
        if self.is_multi_worker:
            print('# {} Processing End, PID: {}'.format(batch, os.getpid()))
        return {'Incomplete Cells List': in_compelet_cells_list,
                'Verification Failed Cells': err_para_cells_list}

    def tray_worker(self, tray_worker_dic: dict) -> (list, list):
        """"""

        curr_batch = tray_worker_dic['batch']
        curr_tray = tray_worker_dic['tray']
        current_batch_static_data = tray_worker_dic['static_df']
        current_batch_tray_path = tray_worker_dic['tray_path']

        if self.is_multi_worker:
            print('# {} Processing Start, PID: {}'.format(curr_batch + '_' + curr_tray, os.getpid()))

        err_para_cells_list = []
        in_compelet_cells_list = []

        tray_name = self.tray_name_adjust(curr_tray)
        current_tray_static_data = current_batch_static_data[current_batch_static_data['Tray ID']
                                                             == tray_name]
        assert current_tray_static_data.shape[0] == self.num_of_cell_in_tray
        current_batch_tray_params_path = current_batch_tray_path
        params_list = os.listdir(current_batch_tray_params_path)
        paras_tray_dic = {}
        for para_tray in iter(params_list):
            # read current batch-tray-parameter
            para_name = self.parameter_name_adjust(para_tray)
            current_para_tray_data = self.read_dynamic_para_tray(current_batch_tray_params_path, para_tray)
            current_para_tray_data.iloc[0, 0] = curr_batch + '-' + para_tray
            paras_tray_dic.update({f'{para_name}': current_para_tray_data})
        cell_no = [k for k in range(1, self.num_of_cell_in_tray + 1)]

        # progress cell_bar parameter setup
        cell_total = len(cell_no)
        text = "#{}-Processing..., PID: {}".format(curr_batch + '_' + curr_tray, os.getpid())
        with tqdm(total=cell_total, desc=text) as cell_bar:
            for cell in iter(cell_no):
                cell_bar.update()
                paras_dic = {}
                err_para_list = []
                curr_cell = curr_batch + '_' + curr_tray + '_' + str(cell)
                # get cell's static data
                static_df = current_tray_static_data.iloc[cell - 1, 1:].to_frame().astype('str', 1)
                static_df_trans = pd.DataFrame(static_df.values.T,
                                               index=static_df.columns,
                                               columns=static_df.index)
                static_df_cp = static_df_trans.copy()
                paras_dic.update({'Static': static_df_cp})
                if len(paras_tray_dic.keys()) < 5:
                    in_compelet_cells_list.append(curr_cell)
                for para in iter(paras_tray_dic.keys()):
                    # verify, such as:   Charge #1          Other
                    #                  End Voltage(mV)   Capacity(mAH)
                    curr_para = para
                    # para extract mode:time, voltage, current, capacity
                    para_df = self.dynamic_para_extract_cell(cell, paras_tray_dic[para])
                    para_ver_flag = self.data_verification(static_df_trans, para_df, para)
                    if (not para_df.empty) and para_ver_flag:
                        paras_dic.update({f'{para}': para_df.copy()})
                        # update paras_dic to cell_dic
                    elif not para_ver_flag:
                        err_para_list.append(curr_cell + '_' + curr_para)
                if err_para_list:
                    err_para_cells_list.append(curr_cell)
                # update cells_dic
                # batch_cells_dic.update({f'{curr_cell}': paras_dic.copy()})
                if self.is_write:
                    write_xlsx_path = os.path.join(self.out_path + '\\organized_data\\xlsx', curr_cell + '.xlsx')
                    if not os.path.exists(write_xlsx_path):
                        with pd.ExcelWriter(write_xlsx_path) as writer:
                            # print(f'Write Cell File: {self.curr_cell}.', end='')
                            for name in iter(paras_dic.keys()):
                                paras_dic[name].to_excel(writer, sheet_name=name)
                    # write cell pickle file
                    write_pickle_path = os.path.join(self.out_path +
                                                     '\\organized_data\\pickle', curr_cell + '.pickle')
                    if not os.path.exists(write_pickle_path):
                        file_operation.save_dic_as_pickle(write_pickle_path, paras_dic)
        if self.is_multi_worker:
            print('# {} Processing End, PID: {}'.format(curr_batch + '_' + curr_tray, os.getpid()))
        return {'Incomplete Cells List': in_compelet_cells_list,
                'Verification Failed Cells': err_para_cells_list}


class DataProcessor():
    """"""
    def __init__(self,
                 organized_file_path: str,
                 data_set_path: str,
                 param_mod_dic: dict,
                 file_type: str = 'pickle',
                 is_multi_worker: bool = True):
        """
        param_mode_dic = {
        'Static': ['Form-OCV #1', ],
        'Charge #1': ['time', 'voltage'],
        'Charge #2': [],
        'Charge #3': [],
        'Discharge': [],
        'Charge #4': []
        }
        """
        # save input
        self.organized_file_path = organized_file_path
        self.data_set_path = data_set_path
        self.param_mod_dic = param_mod_dic
        self.is_multi_worker = is_multi_worker
        self.file_type = file_type

        self. num_processes = mp.cpu_count() - 1
        self.npy_file_path = os.path.join(self.organized_file_path + '\\npy')
        self.load_func = None
        self.used_key = []

    def cell_data_selection(self, data_dic: dict) -> dict:
        """"""
        cells_list = list(data_dic.keys())
        cells_dic = {}
        for use_key in iter(self.param_mod_dic.keys()):
            if self.param_mod_dic[use_key]:
                self.used_key.append(use_key)
        list_extracted = copy.deepcopy(cells_list)
        for cell_name in iter(cells_list):
            cell_dic = {}
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
                    para_list = self.param_mod_dic[used]
                    para = data_dic[cell_name][used][[x for x in iter(para_list)]]
                    cell_dic.update({f'{used}': para})
                cells_dic.update({f'{cell_name}': cell_dic})
        return cells_dic

    @staticmethod
    def data_convert(cells_dic: dict):
        """"""
        cell_list = list(cells_dic.keys())
        dic_extracted_converted = {}
        for cell in iter(cell_list):
            temp_data_dic = cells_dic[cell]
            for key in iter(temp_data_dic.keys()):
                if key == 'Static':
                    temp_data_dic[key] = temp_data_dic[key].astype('float32', 1)
                else:
                    temp_data_dic[key] = temp_data_dic[key].iloc[1:, :].astype('float32', 1)
            dic_extracted_converted.update({f'{cell}': temp_data_dic})
        return dic_extracted_converted

    def data_processing(self):
        """"""
        data = self.data_load_select_convert()
        self.ch1_data_clean_and_interpolation(data)  # specified for ch1_formocv
        self.data_divide()

    def data_load_select_convert(self):
        """"""
        file_list = self.get_file_path_list()

        # set load func
        if self.file_type == 'pickle':
            self.load_func = file_operation.load_dic_in_pickle
        elif self.file_type == 'xlsx':
            self.load_func = file_operation.read_xlsx_all_sheet
        else:
            assert self.load_func

        print('Load, Select, Convert Start.')
        # load, selection and convert
        if self.is_multi_worker:
            with mp.Pool(processes=self.num_processes) as pool:
                res = pool.map(self.load_select_convert_wrapper, file_list)
        else:
            res = []
            with tqdm(total=len(file_list)) as bar:
                bar.set_description('Loading ,Selecting and Concerting ....')
                for j in file_list:
                    bar.update()
                    temp_data_dic = self.load_select_convert_wrapper(j)
                    res.append(temp_data_dic)
        print('Load, Select, Convert End.')

        # transform res to dict
        res = self.res_to_dic(res)
        return res

    def get_file_path_list(self) -> list:
        """"""
        # get file path list
        files_path = os.path.join(self.organized_file_path, self.file_type)
        assert os.path.exists(files_path)
        files_list = os.listdir(files_path)
        files_path_list = []
        for i in iter(files_list):
            temp_path = os.path.join(files_path, i)
            files_path_list.append(temp_path)
        return files_path_list

    def load_select_convert_wrapper(self, files_path: str, ):
        """"""
        # start progress
        raw_data = self.load_func(files_path)
        selected_data = self.cell_data_selection(raw_data)
        converted_data = self.data_convert(selected_data)
        return converted_data

    def data_divide(self, ratio=0.2):
        """"""
        file_list = os.listdir(self.npy_file_path)
        selected_samples = file_list
        num_of_samples = len(selected_samples)
        random.shuffle(selected_samples)
        shuffled_sample_list = selected_samples

        # calculate number of sample in different dataset
        val_num = math.floor(num_of_samples * ratio)
        test_num = math.floor(num_of_samples * ratio)
        train_num = num_of_samples - val_num - test_num

        # start dividing
        assert (val_num + test_num + train_num) == num_of_samples
        val_list = shuffled_sample_list[0: val_num]
        test_list = shuffled_sample_list[val_num: val_num + test_num]
        train_list = shuffled_sample_list[val_num + test_num: val_num + test_num + train_num]
        divide_dic = {
            'train': train_list,
            'val': val_list,
            'test': test_list
        }
        # List of Tuple[(src, dst), ...]
        file_copy_list = self.get_copy_src_dst_list(divide_dic)
        print('Diving Start.')
        if self.is_multi_worker:
            with mp.Pool(processes=self.num_processes) as pool:
                pool.starmap(shutil.copy, file_copy_list)
        else:
            with tqdm(total=len(file_copy_list)) as bar:
                bar.set_description('Dividing ....')
                for j in file_copy_list:
                    bar.update()
                    shutil.copy(j[0], j[1])
        print('Diving End.')

    def get_copy_src_dst_list(self, div_dic: dict) -> list:
        """"""
        out = []
        for key in iter(div_dic.keys()):
            temp_src_fold = self.npy_file_path
            temp_dst_fold = os.path.join(self.data_set_path, key)
            for file in iter(div_dic[key]):
                temp_src_file_path = os.path.join(temp_src_fold, file)
                temp_dst_file_path = temp_dst_fold
                temp_tup = (temp_src_file_path, temp_dst_file_path)
                out.append(temp_tup)
        return out

    @staticmethod
    def res_to_dic(res: list) -> dict:
        """"""
        out_dic = {}
        for i in res:
            temp = i
            out_dic.update(temp)
        return out_dic

    # method for charge#1 voltage curve
    def ch1_data_clean_and_interpolation(self, data_dic):
        """"""
        #
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
        ch1_df_list = []
        with tqdm(total=len(list(data_dic.keys()))) as bar:
            bar.set_description('Charge #1 Data Concatenating....')
            for key in iter(data_dic):
                bar.update()
                temp_df = data_dic[key][para_name]
                ch1_df_list.append(temp_df)
        first_df = pd.concat(ch1_df_list, axis=1)
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

        #
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
        a_index = a_sum < (a.shape[0] / 2)  # True for del col
        del_index = np.where(a_index == True)
        align_df_col_list = list(align_df.columns)

        #
        index_to_col_list = []
        for index in iter(del_index[1]):
            temp_col = align_df_col_list[index]
            index_to_col_list.append(temp_col)
        align_df = align_df.drop(index_to_col_list, axis=1)
        col_list = list(align_df.columns)

        #
        if self.is_multi_worker:
            mp_list = []
            for col in col_list:
                temp_tup = (col, data_dic, align_df, self.npy_file_path)
                mp_list.append(temp_tup)

            with mp.Pool(processes=self.num_processes) as pool:
                pool.starmap(self.ch1_save_npy, mp_list)
        else:
            with tqdm(total=len(col_list)) as w_bar:
                w_bar.set_description('Charge #1 Data Writing....')
                for t, col in enumerate(iter(col_list)):
                    w_bar.update()
                    str_split = col.split('_')
                    cell_name = str_split[0] + '_' + str_split[1] + '_' + str_split[2]
                    static_data = data_dic[cell_name]['Static'].values.squeeze(axis=1)
                    dynamic_data = align_df[col].values
                    sample_with_label = np.concatenate([dynamic_data, static_data])
                    file_name = cell_name + '.npy'
                    save_path = os.path.join(self.npy_file_path, file_name)
                    np.save(save_path, sample_with_label)

    @staticmethod
    def ch1_normalization():
        """"""
        pass

    @staticmethod
    def ch1_standardization():
        """"""
        pass

    @staticmethod
    def ch1_save_npy(col: str, data_dic: dict, align_df: pd.DataFrame, tgt_fold_path: str):
        """"""
        str_split = col.split('_')
        cell_name = str_split[0] + '_' + str_split[1] + '_' + str_split[2]
        static_data = data_dic[cell_name]['Static'].values.squeeze(axis=1)
        dynamic_data = align_df[col].values
        sample_with_label = np.concatenate([dynamic_data, static_data])
        file_name = cell_name + '.npy'
        save_path = os.path.join(tgt_fold_path, file_name)
        np.save(save_path, sample_with_label)


class DataSetCreator(Dataset):
    """"""
    def __init__(self, path: str, transform: tuple = None, trans_para: tuple = None) -> None:
        self.file_path = path
        self.list_files = os.listdir(self.file_path)
        self.num_of_samples = len(self.list_files)

        #
        self.transform = None
        self.trans_para = None
        if (transform is not None) and (trans_para is not None):
            assert (type(transform) == tuple) and (type(trans_para) == tuple)
            assert len(transform) == len(trans_para)
            self.transform = transform
            self.trans_para = trans_para

        # join path
        self.file_path_list = []
        for i in iter(self.list_files):
            temp_file_path = os.path.join(self.file_path, i)
            self.file_path_list.append(temp_file_path)

    def __getitem__(self, index):
        data = np.load(self.file_path_list[index])
        temp_data = data[0:-1]
        temp_label = data[-1]
        if (self.transform is not None) and (self.trans_para is not None):
            temp_data = data[0:-1]
            temp_label = data[-1]
            for tran_i, para_i in zip(iter(self.transform), iter(self.trans_para)):
                temp_data = tran_i(temp_data, **(para_i if para_i else {}))
        return {'data': temp_data,
                'label': temp_label}

    def __len__(self):
        return len(self.file_path_list)

    @classmethod
    def creat_dataset(cls,
                      data_path: str,
                      bsz: int = 32,
                      is_shuffle: bool = True,
                      num_of_worker: int = 0,
                      transform: tuple = None,
                      trans_para: tuple = None) -> (DataLoader, int):
        """"""
        assert os.path.exists(data_path)
        data_set = cls(data_path, transform, trans_para)
        batch_data_set = DataLoader(data_set, batch_size=bsz, shuffle=is_shuffle, num_workers=num_of_worker)
        num_of_batch = math.floor(len(os.listdir(data_path)) / bsz)
        return batch_data_set, num_of_batch


# Module Test
if __name__ == '__main__':
    mp.freeze_support()
    # input_data_path = '.\\data\\2600P-01'
    # output_data_path = '.\\data\\2600P-01_DataSet'
    # m_file_organizer = FileOrganizer(input_data_path, output_data_path, is_multi_worker=False)
    # m_file_organizer.file_organize_work_in_tray(is_write=False)

    organized_file_path = '.\\data\\2600P-01_DataSet\\organized_data'
    data_set_path = '.\\data\\2600P-01_DataSet\\data_set'
    param_mode_dic = {
        'Static': ['Form-OCV #1', ],
        'Charge #1': ['time', 'voltage'],
        'Charge #2': [],
        'Charge #3': [],
        'Discharge': [],
        'Charge #4': []
    }
    m_data_processor = DataProcessor(organized_file_path,
                                     data_set_path,
                                     param_mode_dic,
                                     file_type='pickle',
                                     is_multi_worker=True)
    m_data_processor.data_processing()
    sys.exit(0)
