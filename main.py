import os
import sys
import pandas as pd
import data_loader


#  select which data to be use.
# 'static': use static data.
# 'dynamic': use dynamic data.
data_used_mod = {'static', 'dynamic'}

#  path of dataset
cells_data_path = '.\\data\\2600P-01-organized\\cells'

# Load data from folds
train, test = tdl.load_data