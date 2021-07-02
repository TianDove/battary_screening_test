#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# set net loss function
criterion = nn.MSELoss()

def test(model, data_set, tokenizer, num_of_data, device='cpu'):
    """"""
    # set model val mode
    model.to(device=device)
    model.eval()
    test_total_loss = 0
    test_loss_list = []
    start_time = time.time()  # get start time
    with torch.no_grad():
        for j, raw_data in enumerate(data_set):
            # get label and tokenize data
            data_batch = raw_data['data'].to(torch.float32)
            label_batch = raw_data['label'].to(torch.float32) / 1000

            # move data to GPU
            input_batch = data_batch.to(device)
            label_batch = label_batch.to(device)

            # start batch
            out_batch = model(input_batch)
            if len(out_batch.shape) > 1:
                out_batch = out_batch.squeeze(1)

            # calculate batch loss
            temp_test_loss = criterion(out_batch, label_batch).item()
            test_total_loss += temp_test_loss
            test_loss_list.append(temp_test_loss)

    # return test los
    test_total_loss /= num_of_data
    # log val
    print('#' * 89)
    print('| Test Stage | time: {:5.2f}s | test loss {:7.6f} | '
          .format((time.time() - start_time), test_total_loss))
    print('#' * 89)
    return test_total_loss


if __name__ == '__main__':

    import run
    import model_define
    import functional
    import data_loader

    data_set_name = 'val'
    data_path = '.\\data\\2600P-01_DataSet\\dataset\\' + data_set_name

    model_name = '20210625-153946'
    model_path = '.\\runs\\MyModel\\' + model_name + '\\' \
                 + 'PE_fixed_EC_transformer_DC_mlp_linear_100_0.0003652175318648189_0.0021087178504588244.pt'

    # setup tokenizer parameter
    d_model = 32
    is_overlap = False
    t_step = 8
    token_tup = (d_model, is_overlap, t_step)

    # init tokenizer
    tokenizer = functional.Tokenizer(token_tup)
    tokenizer.calculate_expamle_detoken_len(data_path)

    # setup dataset and device
    batch_size = 64
    device = functional.try_gpu()
    dataset, num_of_batch = data_loader.creat_dataset(data_path,
                                                      bsz=batch_size,
                                                      transform=tokenizer.token_wrapper,
                                                      trans_para='token')

    # setup model parameter
    dropout = 0.1
    nhd = 4
    nly = 3
    hid = 512

    checkpoint = torch.load(model_path, map_location=device)
    # define model
    model_dec_mlp_linear = model_define.PE_fixed_EC_transformer_DC_mlp_linear(d_model,
                                                                              tokenizer,
                                                                              nhd=nhd,
                                                                              nly=nly,
                                                                              dropout=dropout,
                                                                              hid=hid)
    # model_dec_mlp_linear_res = model_define.PE_fixed_EC_transformer_DC_mlp_linear_with_Resconnection
    model_dec_mlp_linear.load_state_dict(checkpoint['model_state_dict'])
    # test
    test(model_dec_mlp_linear, dataset, tokenizer, num_of_batch, device)
