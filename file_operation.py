import os
import sys

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
