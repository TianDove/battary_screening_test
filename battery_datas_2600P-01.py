"""
battery_data_2600P-01
"""


import os
import pandas
import numpy


class File_Organizer(object):
    """"""

    def __init__(self, unsorted_path):
        # 公开属性
        self.unsorted_data_fold_path = unsorted_path
        self.organized_data_fold_path = os.path.join(os.path.dirname(unsorted_path), '2600P-01-organized')
        self.data_type = ('dynamic', 'static')

        # 私有属性（）

    def dynamic_organizer_file(self):
        """"""
        dynamic_fold_path = os.path.join(self.unsorted_data_fold_path, self.data_type[0])
        target_dynamic_fold_path = os.path.join(self.organized_data_fold_path, self.data_type[0])
        dynamic_batch_fold_list = os.listdir(dynamic_fold_path)

        if not os.path.exists(target_dynamic_fold_path):
            os.mkdir(target_dynamic_fold_path)
            print('Fold: \" {} \" created success.'.format(target_dynamic_fold_path))
        else:
            print('Fold: \" {} \" already exists.'.format(target_dynamic_fold_path))

        for batch_fold_name in dynamic_batch_fold_list:
            target_batch_fold_path = os.path.join(target_dynamic_fold_path, batch_fold_name)
            if not os.path.exists(target_batch_fold_path):
                os.mkdir(target_batch_fold_path)
                print('Fold: \" {} \" created success.'.format(target_batch_fold_path))
            else:
                print('Fold: \" {} \" already exists.'.format(target_batch_fold_path))

            dynamic_tray_path = os.path.join(dynamic_fold_path, batch_fold_name)
            dynamic_tray_fold_list = os.listdir(dynamic_tray_path)
            for tray_fold_name in dynamic_tray_fold_list:
                if len(tray_fold_name) <= 4:
                    target_tray_fold_name = '0' * (5-len(tray_fold_name)) + tray_fold_name
                else:
                    target_tray_fold_name = tray_fold_name
                target_tray_fold_path = os.path.join(target_batch_fold_path, target_tray_fold_name)
                if not os.path.exists(target_tray_fold_path):
                    os.mkdir(target_tray_fold_path)
                    print('Fold: \" {} \" created success.'.format(target_tray_fold_path))
                else:
                    print('Fold: \" {} \" already exists.'.format(target_tray_fold_path))
                dynamic_data_path = os.path.join(dynamic_tray_path, tray_fold_name)
                dynamic_data_list = os.listdir(dynamic_data_path)
                for data_name in dynamic_data_list:
                    source_data_name = data_name
                    source_data_name_split = os.path.splitext(source_data_name)
                    source_data_path = os.path.join(dynamic_data_path, source_data_name)
                    source_data_csv_file = pandas.read_csv(source_data_path)
                    if len(source_data_name_split[0]) <= 6:
                        target_data_name = '0' * (7 - len(source_data_name_split[0])) + source_data_name_split[0] + \
                                                   source_data_name_split[1]
                        target_data_path = os.path.join(target_tray_fold_path, target_data_name)
                        if not os.path.exists(target_data_path):
                            source_data_csv_file.to_csv(target_data_path)
                            print('Source:{} -- Target:{} organized success.'.format(source_data_name, target_data_name))
                        else:
                            print('File: \" {} \" already exists.'.format(target_data_path))
        return 0

    def static_organizer_file(self):
        """"""
        static_data_path = os.path.join(self.unsorted_data_fold_path, self.data_type[1])
        static_data_name_list = os.listdir(static_data_path)
        source_file_path = static_data_path
        target_file_path = os.path.join(self.organized_data_fold_path, self.data_type[1])

        if not os.path.exists(target_file_path):
            os.mkdir(target_file_path)
            print('Fold: \" {} \" created success.'.format(target_file_path))
        else:
            print('Fold: \" {} \" already exists.'.format(target_file_path))

        for static_file_name in static_data_name_list:
            source = os.path.join(source_file_path, static_file_name)
            source_name = os.path.basename(source)
            target_name = os.path.splitext(source_name)[0] + '.csv'
            target = os.path.join(target_file_path, target_name)
            if not os.path.exists(target):
                self.xlsx_to_csv(source, target)
                print('File: \" {} \" organized success.'.format(target))
            else:
                print('File: \" {} \" already exists.'.format(target))
            continue
        return 0

    def xlsx_to_csv(self, source, target):
        """"""
        xlsx_file = pandas.read_excel(source)
        if source.endswith('.xlsx'):
            xlsx_file.to_csv(target)
        return 0


class Data_Pre_Processor(object):
    """"""
    def __init__(self, organized_path):
        # 公开属性
        self.organized_fold_path = organized_path
        self.data_type = ('dynamic', 'static')
        self.current_batch_ID = ''
        self.current_tray_ID = ''
        self.current_data_ID = ''
        self.current_cell_ID = ''
        self.number_of_cell_in_a_tray = 256
        self.cells_dataframe = pandas.DataFrame()  # 所有电池数据
        # 私有属性（）

    def data_process_static_and_dynamic(self):
        """"""
        # 不同类型数据文件夹路径(静态，动态)
        dynamic_fold_path = os.path.join(self.organized_fold_path, self.data_type[0])
        static_fold_path = os.path.join(self.organized_fold_path, self.data_type[1])

        temp_cells_dic = {}
        temp_cell_dic = {'static parameters': pandas.Series(),
                         'dynamic parameters': pandas.DataFrame()
                         }
        temp_dynamic_parameter_dic = {'1-charge': pandas.DataFrame(),
                                      '2-charge': pandas.DataFrame(),
                                      '3-charge': pandas.DataFrame(),
                                      '4-discharge': pandas.DataFrame(),
                                      '5-charge': pandas.DataFrame(),
                                      }
        temp_data_dic = {'time(min)': pandas.Series(),
                         'voltage(V)': pandas.Series(),
                         'current(mA)': pandas.Series(),
                         'capacity(mAh)': pandas.Series()
                         }
        # 不同批次动态和静态数据文件夹路径
        dynamic_batch_list = os.listdir(dynamic_fold_path)
        for batch in dynamic_batch_list:
            self.current_batch_ID = batch
            static_data_in_current_batch_path = os.path.join(static_fold_path, batch + '.csv')
            static_data_in_current_batch = pandas.read_csv(static_data_in_current_batch_path, low_memory=False)
            dynamic_data_current_batch_fold_path = os.path.join(dynamic_fold_path, self.current_batch_ID)
            dynamic_data_tray_list_in_current_batch = os.listdir(dynamic_data_current_batch_fold_path)
            for tray in dynamic_data_tray_list_in_current_batch:
                self.current_tray_ID = 'C0000' + tray
                dynamic_data_current_tray_fold_path = os.path.join(dynamic_data_current_batch_fold_path, tray)
                dynamic_data_file_list_in_current_tray = os.listdir(dynamic_data_current_tray_fold_path)
                current_tray_static_data = static_data_in_current_batch[static_data_in_current_batch['Tray ID'] == self.current_tray_ID]
                for data in dynamic_data_file_list_in_current_tray:
                    self.current_data_ID = data[6]
                    dynamic_data_current_data_path = os.path.join(dynamic_data_current_tray_fold_path, data)
                    dynamic_data_in_current_tray = pandas.read_csv(dynamic_data_current_data_path, low_memory=False)
                    dynamic_data_time_in_current_tray = dynamic_data_in_current_tray.iloc[:, 1]
                    voltage_index_range = numpy.arange(2, self.number_of_cell_in_a_tray + 2, dtype='int32').tolist()
                    for cell in voltage_index_range:
                        static_idx = cell - 2
                        cell_static = current_tray_static_data.iloc[static_idx, :]

                        voltage_idx = cell
                        cell_voltage = dynamic_data_in_current_tray.iloc[:, voltage_idx]

                        currnet_idx = voltage_idx + self.number_of_cell_in_a_tray
                        cell_current = dynamic_data_in_current_tray.iloc[:, currnet_idx]

                        capacity_idx = currnet_idx + self.number_of_cell_in_a_tray
                        cell_capacity = dynamic_data_in_current_tray.iloc[:, capacity_idx]

                        temp_data_dic['time(min)'] = dynamic_data_time_in_current_tray
                        temp_data_dic['voltage(V)'] = cell_voltage
                        temp_data_dic['current(mA)'] = cell_current
                        temp_data_dic['capacity(mAh)'] = cell_capacity

                        if self.current_data_ID == 1:
                            temp_dynamic_parameter_dic['1-charge'] = temp_data_dic
                    return 0


# test
m_raw_data_fold = '2600P-01-unsorted'  # 原始数据文件夹名称
m_organized_data_fold = '2600P-01-organized'
m_data_fold = os.path.join(os.getcwd(), 'data')

m_raw_data_fold_path = os.path.join(m_data_fold, m_raw_data_fold)
m_organized_data_fold_path = os.path.join(m_data_fold, m_organized_data_fold)

#m_file_organizer = File_Organizer(m_raw_data_fold_path)
#m_file_organizer.static_organizer_file()
#m_file_organizer.dynamic_organizer_file()

m_data_pre_processor = Data_Pre_Processor(m_organized_data_fold_path)
m_data_pre_processor.data_process_static_and_dynamic()

