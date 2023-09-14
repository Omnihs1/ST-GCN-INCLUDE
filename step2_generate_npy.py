import numpy as np
from numpy.lib.format import open_memmap
import pickle
import json
import os
# import torch
# import torch.nn as nn
from pathlib import Path
# Feeder đọc file json và label map để tạo ra npy và label
class Feeder_include(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in include-skeleton dataset
    Arguments:
        data_path: the path to json data
        label_path: the path to label
        window_size: The length of the output sequence
    """

    def __init__(self,
                 data_path,
                 label_path,
                 window_size=-1):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.load_data()

    def load_data(self):
        self.sample_name = os.listdir(self.data_path)
        # print(f"all files: {self.sample_name}")
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)
        
        # print(label_info)
        self.label_map = label_info

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 2  #channel
        self.T = 80  #frame
        self.V = 25  #joint
        self.M = 1 #person

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, index):
        # output shape (C, T, V)
        # get data
        sample_name = self.sample_name[index]
        # print(f"File name {sample_name}")
        sample_path =  Path(self.data_path) / f"{sample_name}"
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.M))
        for frame_info in video_info:
            frame_index = frame_info['frame_index']
            if frame_index > self.window_size:
                break
            pose = frame_info['pose']
            label = frame_info['label']
            data_numpy[0, frame_index-1, :,self.M-1] = pose[0::2]
            data_numpy[1, frame_index-1, :,self.M-1] = pose[1::2]

        # get & check label index
        label = self.label_map[label]
        # print(f"Numpy {data_numpy}")
        # print(f"Label {label}")
        return data_numpy, label

def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        max_frame=80):

    feeder = Feeder_include(
        data_path=data_path,
        label_path=label_path,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 2, max_frame, 25, 1)
    )

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

gendata("save_data_train", "label.json", "npy_train.npy", "label_train.pickle")