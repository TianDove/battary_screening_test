#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import os
import random
from tqdm import tqdm
import sys
import shutil


def smaller_data_set_for_debug(path, rate=0.1):
    if os.path.exists(path):
        path_xlsx = os.path.join(path, 'xlsx')
        path_pickle = os.path.join(path, 'pickle')
        xlsx_file_list = os.listdir(path_xlsx)
        pickle_file_list = os.listdir(path_pickle)
        if len(xlsx_file_list) == len(pickle_file_list):
            random.shuffle(xlsx_file_list)
            num_of_file = len(xlsx_file_list)
            num_to_del = num_of_file - int(num_of_file * rate)
            with tqdm(total=num_to_del) as bar:
                bar.set_description('Deleting File....')
                for i, file in enumerate(iter(xlsx_file_list)):
                    bar.update()
                    if i == num_to_del - 1:
                        break
                    cell_name = os.path.splitext(file)[0]
                    file_path_xlsx = os.path.join(path_xlsx, cell_name + '.xlsx')
                    file_path_pickle = os.path.join(path_pickle, cell_name + '.pickle')
                    DeleteFile(file_path_xlsx)
                    DeleteFile(file_path_pickle)


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


if __name__ == '__main__':
    path = '.\\data\\2600P-01_DataSet'
    # smaller_data_set_for_debug(path)
    sys.exit(0)
