
import sys
import data_loader
import torch
import torch.utils.data
import functional

#  select which data to be use.
stage_list = ['1-charge', '2-charge', '3-charge', '4-discharge', '5-charge', 'static_data']
param_list = ['time', 'voltage', 'current', 'capacity']
#  1 - use, 0 - no use
param_mod_dic = {'1-charge': {'is_used': 1, 'index': ['time', 'voltage']},
                 '2-charge': {'is_used': 0, 'index': []},
                 '3-charge': {'is_used': 0, 'index': []},
                 '4-discharge': {'is_used': 0, 'index': []},
                 '5-charge': {'is_used': 0, 'index': []},
                 'static_data': {'is_used': 1, 'index': ['Form-OCV #1']}}
running_mod = ['running', 'debug']
epochs = 10
batch_size = 32
num_of_worker = 0
#  path of dataset
cells_data_path = '.\\data\\2600P-01-organized\\cells'
cells_divided_data_path = '.\\data\\2600P-01-organized\\organized_dataSet'

if __name__ == '__main__':
    # Load data from folds
    m_data_loader = data_loader.DataLoader(cells_data_path, cells_divided_data_path, param_mod_dic,
                                           running_mod='running')
    train, val, test = m_data_loader.dataDividing()
    val_dataset = data_loader.cellDataSet(m_data_loader.getSampleList())
    val_dataset_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   num_workers=num_of_worker, drop_last=True)
    for epoch in range(0, epochs):
        for data_list in val_dataset_iter:
            data = m_data_loader.dataLoad(data_list)
            processed_data = functional.ch1_formocv_pre_pro(data)
    sys.exit(0)
