import pandas as pd
import numpy as np
import scipy.io as sio
import torch

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

def save_obj_dic_sio(path, list_data):
    sio.savemat(path, list_data)

def load_obj_dic_sio(path):
    temp = sio.loadmat(path)
    return temp

def mat_to_dic(mat_data):
    pass

def ch1_formocv_pre_pro(datas):
    batch_tensor = torch.zeros(len(datas), 1, 160)
    label = torch.zeros(len(datas), 1, 2)
    counter = 0
    for data in iter(datas):
        try:
            time = datas[data]['1-charge']['time']
            voltage = datas[data]['1-charge']['voltage']
            form_ocv = float(datas[data]['static_data']['Form-OCV #1'])
            voltage_div = sequenceDiff(voltage).div(sequenceDiff(time))
            # batch_tensor[counter, :, :] = torch.tensor(voltage_div.values)
            label_form_ocv = torch.tensor([3600.0 - form_ocv, form_ocv - 3400.0], dtype=torch.float16)
            # label[counter, :, :] = label_form_ocv
            counter += 1
            print(data, voltage_div.shape[0])
        except KeyError:
            x = 1


