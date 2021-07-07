#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import os
import sys
import functional
import pandas as pd
from tqdm import tqdm
import file_operation
import multiprocessing as mp


class File_Organizer():
    """"""
    def __init__(self, fold_path, fold_name):
        """"""
        self.fold_path = fold_path
        self.fold_name = fold_name
        self.cell_fold_name = '2600P-01_DataSet'
        self.num_of_cell_in_tray = 256
        self.output_fold_path = os.path.join(self.fold_path, self.cell_fold_name)
        self.input_fold_path = os.path.join(self.fold_path, self.fold_name)
        self.curr_batch = ''
        self.curr_tray = ''
        self.curr_cell = ''
        self.curr_para = ''
        self.cell_ver_err_dic = {}
        self.voltage_index = [x for x in range(1, self.num_of_cell_in_tray + 1)]
        self.current_index = [x for x in range(self.num_of_cell_in_tray + 1, self.num_of_cell_in_tray*2 + 1)]
        self.capacity_index = [x for x in range(self.num_of_cell_in_tray*2 + 1, self.num_of_cell_in_tray*3 + 1)]
        self.write_flag = False

    def make_file_dir(self):
        """"""
        if not os.path.exists(self.input_fold_path):
            print(f'Path Not Exist:{self.input_fold_path}.')
            sys.exit(0)
        if not os.path.exists(self.output_fold_path):
            os.mkdir(self.output_fold_path)
            os.mkdir(self.output_fold_path + '\\xlsx')
            os.mkdir(self.output_fold_path + '\\pickle')
            print(f'Make Dir:{self.output_fold_path}.')
            return True
        else:
            print(f'Path Exist:{self.output_fold_path}.')
            sys.exit(0)

    def tray_name_adjust(self, tray):
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

    def parameter_name_adjust(self, para):
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

    def dynamic_para_extract(self, para_data):
        """"""
        para_dic = {}
        # time
        time = para_data.iloc[:, 0]
        para_dic.update({'time': time})
        # voltage
        voltage = para_data.iloc[:, self.voltage_index]
        para_dic.update({'voltage': voltage})
        # current
        current = para_data.iloc[:, self.current_index]
        para_dic.update({'current': current})
        # capacity
        capacity = current = para_data.iloc[:, self.capacity_index]
        para_dic.update({'capacity': capacity})
        return para_dic

    def dynamic_para_extract_cell(self, cell_no, para_data):
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

    def data_verification(self, static_df, para_df, ver_type):
        """Un-used"""
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

    def read_static_data(self, batch):
        """"""
        current_batch_static_data_path = os.path.join(self.input_fold_path + '\\static', batch)
        if os.path.getsize(current_batch_static_data_path):
            # print(f'Reading File {batch}.', end='\r')
            static_data = pd.read_csv(open(current_batch_static_data_path, 'rb'), low_memory=False)
            # print(f'Reading Completed {batch}.', end='\r')
            return static_data
        else:
            print('Static File Size = 0, Error.')
            sys.exit(0)

    def read_dynamic_para_tray(self, para_path, para_name):
        """"""
        current_para_path = os.path.join(para_path, para_name)
        if os.path.getsize(current_para_path):
            current_para_data = pd.read_csv(current_para_path)
            return current_para_data
        else:
            print('Dynamic File Size = 0, Error.')
            sys.exit(0)

    def cell_no_to_tray_cell_no(self, cell_no):
        cell_no = int(cell_no)
        tray_cell_no = -1
        alp = ('A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P')
        res = (cell_no - 1) % 16
        times = int((cell_no - 1) / 16)
        tray_cell_no = alp[times] + str(res + 1)
        return tray_cell_no

    def file_organize(self, write: bool = False, num_of_worker: int = None):
        """
        2600P-01 文件夹结构：
        2600P-01
        |
        --dynamic -- 210301-1 -- 0397 -- 0397-1.csv
        |         |          |       |
        |         |          |       ...
        |         |          ...
        |         ...
        |
        _static  -- 210301-1.xlsx
                |
                ...
        """
        self.write_flag = write
        # self.make_file_dir()
        # dynamic data path and batch list
        dynamic_data_path = os.path.join(self.input_fold_path, 'dynamic')
        dynamic_batch_list = os.listdir(dynamic_data_path)

        # static data path and batch list
        static_data_path = os.path.join(self.input_fold_path, 'static')
        static_batch_list = os.listdir(static_data_path)

        #
        # cells_pickle_path = os.path.join('.\\data\\cells_dic.pickle')
        # if os.path.exists(cells_pickle_path):
        #    print(f'Pickle Save Path Exist.')
        #    sys.exit(0)

        # organize start
        print(f'Input Raw Data Path:{self.input_fold_path}')
        print(f'Output Cell Data Path:{self.output_fold_path}')
        # progress bar
        # batch_bar = tqdm(len(dynamic_batch_list))
        # dic save all cell's parameters
        # | Cell Name(batch_tray_CellNo)
        # |
        # | Charge #1, Charge #2, Charge #3, Discharge, Charge #4, Static
        # |            |          |          |          |          |
        # |            ...        ...        ...        ...         ...
        # |
        # |--- time - voltage - current - capacity
        cells_dic = {}
        err_list = []
        for batch in iter(dynamic_batch_list):
            self.curr_batch = batch
            if (batch + '.csv') in static_batch_list:
                #  read current batch static file
                # print(f'Reading File {batch}.', end='\r')
                current_batch_static_data = self.read_static_data(batch + '.csv')
                # print(f'Reading Completed {batch}.')
                # current batch tray list
                current_batch_tray_path = os.path.join(dynamic_data_path, batch)
                tray_list = os.listdir(current_batch_tray_path)
                # tray progress bar
                # tray_bar = tqdm(len(tray_list))
                for tray in iter(tray_list):
                    tray_name = self.tray_name_adjust(tray)
                    self.curr_tray = tray_name
                    current_tray_static_data = current_batch_static_data[current_batch_static_data['Tray ID']
                                                                         == tray_name]
                    if current_tray_static_data.shape[0] != self.num_of_cell_in_tray:
                        print('Current Tray Static Data Shape Error.')
                        sys.exit(0)
                    current_batch_tray_params_path = os.path.join(current_batch_tray_path, tray)
                    params_list = os.listdir(current_batch_tray_params_path)
                    # dynamic para tray dic
                    paras_tray_dic = {}
                    for para_tray in iter(params_list):
                        # read current batch-tray-parameter
                        para_name = self.parameter_name_adjust(para_tray)
                        # print(f'Reading File {batch}-{tray}-{para_name}.', end='\r')
                        current_para_tray_data = self.read_dynamic_para_tray(current_batch_tray_params_path, para_tray)
                        current_para_tray_data.iloc[0, 0] = self.curr_batch + '-' + para_tray
                        # print(f'Reading File {batch}-{tray}-{para_name} Completed.')
                        paras_tray_dic.update({f'{para_name}': current_para_tray_data})
                    if not paras_tray_dic == {}:
                        # all para of a tray will saved into paras_dic
                        cell_no = [k for k in range(1, self.num_of_cell_in_tray + 1)]
                        # Cell progress Bar
                        with tqdm(total=self.num_of_cell_in_tray) as cell_bar:
                            cell_bar.set_description(f'{self.curr_batch}-{self.curr_tray}-Cell Progress:')
                            for cell in iter(cell_no):
                                paras_dic = {}
                                self.curr_cell = self.curr_batch + '_' + self.curr_tray + '_' + str(cell)
                                # print(f'Processing Cell: {self.curr_cell}.', end=' ')
                                # get cell's static data
                                static_df = current_tray_static_data.iloc[cell - 1, 1:].to_frame().astype('str', 1)
                                static_df_trans = pd.DataFrame(static_df.values.T,
                                                               index=static_df.columns,
                                                               columns=static_df.index)
                                static_df_cp = static_df_trans.copy()
                                paras_dic.update({'Static': static_df_cp})
                                # data verification list
                                para_err_list = []
                                for para in iter(paras_tray_dic.keys()):
                                    # verify, such as:   Charge #1          Other
                                    #                  End Voltage(mV)   Capacity(mAH)
                                    self.curr_para = para
                                    # para extract mode:time, voltage, current, capacity
                                    para_df = self.dynamic_para_extract_cell(cell, paras_tray_dic[para])
                                    para_ver_flag = self.data_verification(static_df_trans, para_df, para)
                                    if (not para_df.empty) and para_ver_flag:
                                        paras_dic.update({f'{para}': para_df.copy()})
                                        # update paras_dic to cell_dic
                                    else:
                                        err_name = self.curr_cell + '_' + para
                                        err_list.append(err_name)
                                        print('Parameter Dictionary Empty.')
                                        # sys.exit(0)
                                # update cells_dic
                                # cells_dic.update({f'{self.curr_cell}': paras_dic.copy()})
                                # print('\r' + f'Processing Cell Complete: {self.curr_cell}.', end='\r')
                                if self.write_flag:
                                    write_path = os.path.join(self.output_fold_path
                                                              + '.\\xlsx', self.curr_cell + '.xlsx')
                                    if not os.path.exists(write_path):
                                        with pd.ExcelWriter(write_path) as writer:
                                            # print(f'Write Cell File: {self.curr_cell}.', end='')
                                            for name in iter(paras_dic.keys()):
                                                paras_dic[name].to_excel(writer, sheet_name=name)
                                    # write cell pickle file
                                    write_pickle_path = os.path.join(self.output_fold_path
                                                                     + '.\\pickle', self.curr_cell + '.pickle')
                                    if not os.path.exists(write_pickle_path):
                                        file_operation.save_dic_as_pickle(write_pickle_path, paras_dic)
                                # Cell tray progress bar
                                cell_bar.update()
                    else:
                        print('Parameter in Tray Dictionary Empty.')
                        # sys.exit(0)
            else:
                print(f'Current Batch:{batch}, Corresponding Static data No Found.')
                sys.exit(0)
        # save cells_dic as pickle
        # print(f'Saving Cells Dic....', end='')
        # functional.save_dic_as_pickle(cells_pickle_path, cells_dic)
        # print('\r' + f'Save Cells Dic Complete.')
        err_sq = pd.Series(err_list)
        err_sq.to_csv('.\\data\\file_err_list.csv')


if __name__ == '__main__':
    N_WORKER = 3
    fold_path = '.\\data'
    fold_name = '2600P-01'
    mp.freeze_support()
    m_file_organizer = File_Organizer(fold_path, fold_name)
    with mp.Pool(processes=N_WORKER,) as pool:
        pass


