import os
current_dir = os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random
from scipy.io import loadmat


def TrainDataset(mark_state, snr, len_mask):
    x = []
    y = []
    for i in range(5):
        if mark_state == 'w':
            mat_file_path = os.path.join(current_dir, '..', f'dataset/Train_{mark_state}_mark/QAM16_{snr}dB_Device{i+1}_mark_len={len_mask}.mat')
        else:
            mat_file_path = os.path.join(current_dir, '..', f'dataset/Train_{mark_state}_mark/QAM16_{snr}dB_Device{i+1}.mat')
        dataset = loadmat(mat_file_path)
        data = dataset['IQDataset']
        label = i*np.ones(data.shape[0],)
        x.append(data)
        y.append(label)
    x = np.concatenate(x, 0)
    y = np.concatenate(y, 0)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 2023)
    return x_train, x_val, y_train, y_val


def TestDataset(mark_state, snr, len_mark):
    x = []
    y = []
    for i in range(5):
        mat_file_path = os.path.join(current_dir, '..', f'dataset/Test_{mark_state}_mark/QAM16_{snr}dB_Device{i+1}_mark_len={len_mark}.mat')
        dataset = loadmat(mat_file_path)
        data = dataset['IQDataset']
        label = i*np.ones(data.shape[0],)
        x.append(data)
        y.append(label)
    x = np.concatenate(x, 0)
    y = np.concatenate(y, 0)
    return x, y
