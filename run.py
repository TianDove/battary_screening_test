#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import time
import torch
import torch.nn as nn

# constant
import functional

BATCH_LOG_INTERVAL = 10
# set net loss function
criterion = nn.MSELoss()


def train(model, optimizer, data_set, tokenizer, current_epoch, num_of_batch, device=None):
    """"""
    assert device is not None
    # set model to train mode
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    batch_loss = 0

    # batch loop
    for batch_index, raw_data in enumerate(data_set):
        start_time = time.time()  # get start time
        # get label and tokenize data
        temp_data = raw_data[:, 0:-1]

        #temp_label = raw_data[:, -1]
        #temp_data_pro, d_mean, d_std = functional.data_preprocessing(temp_data)
        #temp_data_de = functional.data_depreprocessing(temp_data_pro, d_mean, d_std)

        input_batch = tokenizer.token_wrapper(temp_data, mode='token')
        label_batch = raw_data[:, -1] / 1000

        # move data to GPU
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        # start batch
        optimizer.zero_grad()
        out_batch = model(input_batch)
        if len(out_batch.shape) > 1:
            out_batch = out_batch.squeeze(1)

        # calculate batch loss
        loss_batch = criterion(out_batch, label_batch)

        # accumulate loss
        scalar_loss = loss_batch.detach().item()
        batch_loss += scalar_loss
        epoch_loss += scalar_loss

        # update model
        loss_batch.backward()
        optimizer.step()

        # log batch train
        log_interval = BATCH_LOG_INTERVAL
        if (batch_index % log_interval == 0 and batch_index > 0) or batch_index == num_of_batch - 1:
            if batch_index == num_of_batch - 1 :
                log_interval = num_of_batch % log_interval
            cur_loss = batch_loss / log_interval
            elapsed = time.time() - start_time
            print('| '
                  'epoch {:3d} | '
                  '{:5d}/{:5d} batches | '
                  'lr {:07.6f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:7.6f} |'.format(
                current_epoch, batch_index, num_of_batch - 1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval,
                cur_loss))
            batch_loss = 0

    # return epoch loss
    epoch_loss /= num_of_batch
    return epoch_loss


def val(model, data_set, tokenizer, current_epoch, num_of_data, device=None):
    """"""
    assert device is not None
    # set model val mode
    model.eval()
    val_total_loss = 0
    val_loss_list = []
    start_time = time.time()  # get start time
    with torch.no_grad():
        for j, raw_data in enumerate(data_set):
            # get label and tokenize data
            temp_data = raw_data[:, 0:-1]
            input_batch = tokenizer.token_wrapper(temp_data, mode='token')
            label_batch = raw_data[:, -1] / 1000

            # move data to GPU
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            # start batch
            out_batch = model(input_batch)
            if len(out_batch.shape) > 1:
                out_batch = out_batch.squeeze(1)

            # calculate batch loss
            temp_val_loss = criterion(out_batch, label_batch).item()
            val_total_loss += temp_val_loss
            val_loss_list.append(temp_val_loss)

    # return val loss
    val_total_loss /= num_of_data

    # log val
    print('-' * 89)
    print('| end of epoch {:3d} '
          '| val total time: {:5.2f}s '
          '| valid loss {:7.6f} | '
          .format(current_epoch,
                  (time.time() - start_time),
                  val_total_loss))
    print('-' * 89)
    return val_total_loss


def test(model, data_set, tokenizer, num_of_data, device=None):
    """"""
    assert device is not None
    # set model val mode
    model.eval()
    test_total_loss = 0
    test_loss_list = []
    start_time = time.time()  # get start time
    with torch.no_grad():
        for j, raw_data in enumerate(data_set):
            # get label and tokenize data
            temp_data = raw_data[:, 0:-1]
            input_batch = tokenizer.token_wrapper(temp_data, mode='token')
            label_batch = raw_data[:, -1] / 1000

            # move data to GPU
            input_batch = input_batch.to(device)
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

