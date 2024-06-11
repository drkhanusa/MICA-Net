import glob
from algorithm import *
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

__all__ = ['UESTCVideoDataGenerator']


class UESTCVideoDataGenerator(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 data_format='channels_first',
                 batch_size=1,
                 shuffle=True):
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError(f'data_format must be either channels_first or channels_last, got {data_format}')
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.data_format = data_format
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.clips = []
        self.labels = []

        data_annotation = []
        mode = pd.read_csv(self.annotation_file_path, header=None).values
        for i in mode:
            data_annotation.append(i[0])

        for video_file in glob.glob(os.path.join(self.video_dir, '*', '*', '*.mp4')):
            label = int(video_file.split("/")[-2]) - 1
            video_name = video_file.split("/")[-1][:-4]
            if video_name in data_annotation:
                self.clips.append(video_file)
                self.labels.append(label)

        self.classes = 32

        self.indices = np.arange(len(self.clips))


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):

        video_file = self.clips[index]
        X = self.sampler(video_file, sample_id=index)

        if self.to_rgb:
            X = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in X]
        if self.transform is not None:
            X = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in X]

        X = np.array(X)
        X = X.transpose((3,0,1,2))
        X = torch.from_numpy(X).float()

        y = self.labels[index]
        y = np.array(y, int)
        y = torch.from_numpy(y)
        return X, y