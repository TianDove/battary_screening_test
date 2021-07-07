#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang


import os
import sys
import time
import datetime
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm, trange

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

    def get_tray_to_process(self) -> list:
        """
        static_df: pd.DataFrame, batch: str, tray: str, tray_path: str
        """
        batch_list = self.get_batch_to_process()
        temp_tray_list = []
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
        batch_tray_list = self.get_tray_to_process()

        # output
        in_comp_cells = []
        err_para_cells = []

        # start processing
        start_time = time.time()
        if self.is_multi_worker:
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                in_comp_cells, err_para_cells = pool.map(self.tray_worker, batch_tray_list)
            end_time = time.time()
            self.log_file_organie(start_time, end_time, in_comp_cells, err_para_cells)
        else:
            print('Single Processing, Pleas Using :self.file_organize_work_in_batch(), '
                  'with self.is_multi_worker = False.')
            sys.exit(0)


    def file_organize_work_in_batch(self, is_write: bool = True):
        """"""
        self.is_write = is_write
        self.make_file_dir()
        batch_list = self.get_batch_to_process()
        in_comp_cells = []
        err_para_cells = []
        start_time = time.time()
        # use multiprocess or not
        if self.is_multi_worker:
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                in_comp_cells, err_para_cells = pool.map(self.batch_worker, batch_list)
            end_time = time.time()
        else:
            for batch in batch_list:
                temp_in_comp_cells, temp_err_para_cells = self.batch_worker(batch)
                in_comp_cells += temp_in_comp_cells
                err_para_cells += temp_err_para_cells
            end_time = time.time()
        # log
        self.log_file_organie(start_time, end_time, in_comp_cells, err_para_cells)
        print('Logging Finished, All Done.')
        sys.exit(0)

    def log_file_organie(self, start, end, in_comp_cells, err_para_cells):
        """"""
        # logging
        now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(self.out_path, now_time + '_' + 'FileOrganizer_Logging.txt')
        assert os.path.exists(log_file_path)
        consumed_time = end - start
        # res = {'Incomplete Cells': in_comp_cells, 'Verification Failed Cells': err_para_cells}

        seperator = '-' * 89
        file_operation.write_txt(log_file_path, seperator)
        text_data = '| Finish Date: {} | Write Flag: {} | Multiprocess Flag: {} | Time Consume: {} s |'.format(
            now_time, self.is_write, self.is_multi_worker, consumed_time)
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Incomplete Cells List ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in iter(in_comp_cells):
            text_data = i
            file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Verification Failed Cells ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in iter(err_para_cells):
            text_data = i
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
        return in_compelet_cells_list, err_para_cells_list

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
        return in_compelet_cells_list, err_para_cells_list

class DataProcessor():
    """"""
    pass


class DataSetCreator():
    """"""
    pass


# Module Test
if __name__ == '__main__':
    mp.freeze_support()
    input_data_path = '.\\data\\2600P-01'
    output_data_path = '.\\data\\2600P-01_DataSet'
    m_file_organizer = FileOrganizer(input_data_path, output_data_path, is_multi_worker=True)
    m_file_organizer.file_organize_work_in_tray(is_write=False)
