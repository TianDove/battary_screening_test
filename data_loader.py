#
import os
import sys
import scipy.io as sio
import torch.utils.data
import pandas as pd
import random
import math

#
import file_operation
import functional

class cellDataSet(torch.utils.data.Dataset):
    """Pytorch DataSet"""
    def __init__(self, samples_list):
        self.List = samples_list
        self.Len = len(samples_list)

    def __getitem__(self, index):
        return self.List[index]

    def __len__(self):
        return self.Len


class DataLoader(torch.utils.data.Dataset):
    """电池数据预处理"""
    def __init__(self, data_path, divided_path, param_mod_dic, running_mod='running'):
        self.isDebug = running_mod
        self.dataPath = data_path
        self.dividedPath = divided_path
        self.paramModDic = param_mod_dic
        self.calc_diff = False

        print(f'Data Path:{self.dataPath}')
        print(f'Data Path:{self.dividedPath}')
        print(f'Current Running Mode = {self.isDebug}')
        key_list = self.paramModDic.keys()
        for i in iter(key_list):
            temp_dic = self.paramModDic[i]
            print(f'Selected Param--{i}--{temp_dic}')

        # Input parameter test
        self.isClassInputCorrect()
        self.sampleList = os.listdir(self.dataPath)

    def isClassInputCorrect(self):
        """类输入参数检测"""
        if os.path.exists(self.dataPath):
            # get train and test data path
            train_path = self.dividedPath + '\\train'
            test_path = self.dividedPath + '\\test'
            if os.path.exists(self.dividedPath):
                if self.isDebug == 'running':
                    # if running, deleting the dataset and re-dividing the original data
                    file_operation.Delete_File_Dir(self.dividedPath)
                    print(f'Dir Removed:{self.dividedPath}')
                    os.mkdir(self.dividedPath)
                    print(f'Dir Created:{self.dividedPath}')
                    os.mkdir(train_path)
                    print(f'Dir Created:{train_path}')
                    os.mkdir(test_path)
                    print(f'Dir Created:{test_path}')
            else:
                os.mkdir(self.dividedPath)
                print(f'Dir Created:{self.dividedPath}')
                os.mkdir(train_path)
                print(f'Dir Created:{train_path}')
                os.mkdir(test_path)
                print(f'Dir Created:{test_path}')
        else:
            print(f'Dir No Found:{self.dataPath}')
            sys.exit(0)

    def dataDividing(self, ratio=0.1):
        """数据集划分"""
        if not self.isDebug:
            selected_samples = self.isUsedParamExist()
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
                print(f'Data Dividing Completed.')
                return train_list, val_list, test_list
            else:
                print(f'Sample Num Validation Error.')
                sys.exit(0)
        else:
            train_list = functional.load_list_np('.\\data\\train.npy')
            val_list = functional.load_list_np('.\\data\\val.npy')
            test_list = functional.load_list_np('.\\data\\test.npy')
            return train_list, val_list, test_list
            
    def dataLoad(self, path_list, mode='train', output_mode='list'):
        path_list_iter = iter(path_list)
        data = {}
        print(f'{mode} Data Loading Start.')
        count = 0
        total = len(path_list)
        for i in path_list_iter:
            cell_path = os.path.join(self.dataPath, i)
            cell_dic = self.readCellDataFile(cell_path)
            data.update({f'{i}': cell_dic})
            count += 1
            prgs = count/total
            print('\r' + f'Processing:{prgs * 100}%--{mode} Data Loading--Cell:{i}', end='')
        print('\n' + f'{mode} Data Loading Completed.')
        return data

    def readCellDataFile(self, file_path):
        """读取电池的参数文件"""
        if os.path.exists(file_path):
            cell_dic = {}
            dataList = os.listdir(file_path)
            for j in iter(dataList):
                data_name = os.path.splitext(j)[0]
                if self.paramModDic[data_name]['is_used']:
                    data_path = os.path.join(file_path, j)
                    data_csv = pd.read_csv(data_path)
                    processed_data = self.featureSelection(data_name, data_csv)
                    temp_dic = {f'{data_name}': processed_data}
                    cell_dic.update(temp_dic)
            return cell_dic
        else:
            print(f'File Path No Found: {file_path}')
            sys.exit(0)

    def featureSelection(self, data_name, data):
        """电池参数处理"""
        if data_name != 'static_data':
            time = pd.Series(data.iloc[1:, 1], dtype='float16')
            voltage = pd.Series(data.iloc[1:, 2], dtype='float16')
            current = pd.Series(data.iloc[1:, 3], dtype='float16')
            capacity = pd.Series(data.iloc[1:, 4], dtype='float16')
            d1 = {'time': time, 'voltage': voltage, 'current': current, 'capacity': capacity}
            d2 = {}
            index_iter = iter(self.paramModDic[data_name]['index'])
            for index in index_iter:
                d2.update({index: d1[index]})
            df_d = pd.DataFrame(d2)
            return df_d
        elif data_name == 'static_data':
            df_index = data.iloc[1:, 0].to_list()
            df_data = data.iloc[1:, 1].to_list()
            s = pd.Series(df_data, index=df_index)
            sr_s = s[self.paramModDic[data_name]['index']]
            return sr_s
        else:
            print(f'DataName Error')
            sys.exit(0)

    def isUsedParamExist(self):
        selected_list = self.sampleList.copy()
        total = len(self.sampleList)
        count = 0
        for i in iter(self.sampleList):
            #  display
            count += 1
            prgs = count / total
            print('\r' + f'Progress: {prgs * 100}%' + '----UsedParamExist Checking....', end='')
            cell_path = os.path.join(self.dataPath, i)
            data_list = os.listdir(cell_path)
            for j in range(0, len(data_list)):
                data_list[j] = os.path.splitext(data_list[j])[0]
            for t in iter(self.paramModDic.keys()):
                if self.paramModDic[t]['is_used']:
                    if not (t in data_list):
                        selected_list.remove(i)
        print('\n' + 'UsedParamExist Checking Completed.')
        return selected_list

    def getSampleList(self):
        return self.sampleList

    def featureCheking(self):
        pass
