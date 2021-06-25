#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import time
import torch
import torch.nn as nn


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
        # get label and tokenize data
        temp_data = raw_data[:, 0:-1]
        input_batch = tokenizer.token_wrapper(temp_data, mode='token')
        label_batch = raw_data[:, -1] / 1000

        # move data to GPU
        input_batch = input_batch.cuda(device)
        label_batch = label_batch.cuda(device)

        # start batch
        start_time = time.time()  # get start time
        optimizer.zero_grad()
        out_batch = model(input_batch)
        if len(out_batch.shape) > 1:
            out_batch = out_batch.squeeze(1)

        # calculate batch loss
        loss_batch = criterion(out_batch, label_batch)

        # accumulate loss
        scalar_loss = loss_batch.detach().to(device='cpu').item()
        batch_loss += scalar_loss
        epoch_loss += scalar_loss

        # update model
        loss_batch.backward()
        optimizer.step()

        # log batch train
        log_interval = 50
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = batch_loss / log_interval
            elapsed = time.time() - start_time
            print('| '
                  'epoch {:3d} | '
                  '{:5d}/{:5d} batches | '
                  'lr {:02.2f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} |'.format(
                current_epoch, batch_index, num_of_batch, 0.0,
                elapsed * 1000 / log_interval,
                cur_loss))
            batch_loss = 0

    # return epoch loss
    return epoch_loss / num_of_batch


def val(model, data_set, tokenizer, current_epoch, num_of_data, device=None):
    """"""
    assert device is not None
    # set model val mode
    model.eval()
    val_loss = 0
    start_time = time.time()  # get start time
    with torch.no_grad():
        for j, raw_data in enumerate(data_set):
            # get label and tokenize data
            temp_data = raw_data[:, 0:-1]
            input_batch = tokenizer.token_wrapper(temp_data, mode='token')
            label_batch = raw_data[:, -1] / 1000

            # move data to GPU
            input_batch = input_batch.cuda(device)
            label_batch = label_batch.cuda(device)

            # start batch
            out_batch = model(input_batch)
            if len(out_batch.shape) > 1:
                out_batch = out_batch.squeeze(1)

            # calculate batch loss
            val_loss += criterion(out_batch, label_batch).to(device='cpu').item()

    # log val
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          .format(current_epoch, (time.time() - start_time), val_loss / num_of_data))
    print('-' * 89)
    return val_loss


def test(model, data_set, tokenizer, num_of_data, device=None):
    """"""
    assert device is not None
    # set model val mode
    model.eval()
    test_loss = 0
    start_time = time.time()  # get start time
    with torch.no_grad():
        for j, raw_data in enumerate(data_set):
            # get label and tokenize data
            temp_data = raw_data[:, 0:-1]
            input_batch = tokenizer.token_wrapper(temp_data, mode='token')
            label_batch = raw_data[:, -1] / 1000

            # move data to GPU
            input_batch = input_batch.cuda(device)
            label_batch = label_batch.cuda(device)

            # start batch
            out_batch = model(input_batch)
            if len(out_batch.shape) > 1:
                out_batch = out_batch.squeeze(1)

            # calculate batch loss
            test_loss += criterion(out_batch, label_batch).to(device='cpu').item()

    # log val
    print('-' * 89)
    print('| Test Stage | time: {:5.2f}s | valid loss {:5.2f} | '
          .format((time.time() - start_time), test_loss / num_of_data))
    print('-' * 89)
    return test_loss