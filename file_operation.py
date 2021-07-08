#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import os
import shutil
import pandas as pd
import pickle
import sys
import openpyxl


def DeleteFile(strFileName):
    """ 删除文件 """
    fileName = str(strFileName)
    if os.path.isfile(fileName):
        try:
            os.remove(fileName)
        except:
            pass


def Delete_File_Dir(dirName, flag = True):
    """ 删除指定目录，首先删除指定目录下的文件和子文件夹，然后再删除该文件夹 """
    if flag:
        dirName = str(dirName)
        """ 如何是文件直接删除 """
    if os.path.isfile(dirName):
        try:
            os.remove(dirName)
        except:
            pass
    elif os.path.isdir(dirName):
        """ 如果是文件夹，则首先删除文件夹下文件和子文件夹，再删除文件夹 """
        for item in os.listdir(dirName):
            tf = os.path.join(dirName,item)
            Delete_File_Dir(tf, False)
            """ 递归调用 """
        try:
            os.rmdir(dirName)
        except:
            pass


def move_file_to_fold(src, dst, file_list):
    if os.path.exists(src) and os.path.exists(dst):
        list_iter = iter(file_list)
        for file in list_iter:
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            shutil.move(src_file, dst_file)


def read_txt(file_path):
    assert os.path.exists(file_path)
    data = None
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


def write_txt(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            print(data, file=file)
    else:
        with open(file_path, 'a', encoding='utf-8') as file:
            print(data, file=file)
        

def xlsx_to_csv(src_path, out_path):
    """将目标文件夹下的xlsx文件转换为csv文件并存储到输出文件夹"""
    xlsx_list = os.listdir(src_path)
    for i in iter(xlsx_list):
        file_path = os.path.join(src_path, i)
        xlsx = pd.read_excel(file_path)
        target_path = os.path.join(out_path, os.path.splitext(i)[0] + '.csv')
        xlsx.to_csv(target_path)


def save_dic_as_pickle(target_path, data_dic):
    """保存为pickle文件"""
    if not os.path.exists(target_path):
        with open(target_path, 'wb') as f:
            pickle.dump(data_dic, f)
    else:
        print(f'Path Exist: {target_path}')
        sys.exit(0)


def load_dic_in_pickle(source_path):
    """读取pickle文件"""
    if os.path.exists(source_path):
        file_name = os.path.basename(source_path)
        file_name = os.path.splitext(file_name)[0]
        dic = {f'{file_name}': {}}
        with open(source_path, 'rb') as f:
            temp_dic = pickle.load(f)
            dic[f'{file_name}'].update(temp_dic)
            return dic
    else:
        print(f'Path Not Exist: {source_path}')
        sys.exit(0)


def write_to_xlsx(target_path, data_dic):
    """"""
    if not os.path.exists(target_path):
        with pd.ExcelWriter(target_path) as writer:
            for name in iter(data_dic.keys()):
                data_dic[name].to_excel(writer, sheet_name=name)
    else:
        print(f'Path Exist: {target_path}')
        sys.exit(0)


def read_xlsx_all_sheet(source_path: str) -> dict:
    """"""
    assert os.path.exists(source_path)
    file_name = os.path.basename(source_path)
    file_name = os.path.splitext(file_name)[0]
    wb = openpyxl.load_workbook(source_path)
    sheets = wb.sheetnames
    dic = {f'{file_name}': {}}
    for sheet in sheets:
        temp_sheet_df = pd.read_excel(source_path, sheet_name=sheet)
        temp_dic = {f'{sheet}': temp_sheet_df}
        dic[f'{file_name}'].update(temp_dic)
    return dic


if __name__ == '__main__':
    sys.exit(0)
