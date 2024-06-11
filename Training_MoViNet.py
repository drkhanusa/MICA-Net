import albumentations as A
from algorithm import *
from six.moves import urllib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset import UESTCVideoDataGenerator
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
from MoViNet.movinets.models import MoViNet
from MoViNet.movinets.config import _C



def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())

def evaluate(model, data_load, loss_val):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')


if __name__ == '__main__':
    fold = './UESTC-MMEA-CL/data/video'
    train_annotations = './UESTC-MMEA-CL/data/train3.txt'
    val_annotations = './UESTC-MMEA-CL/data/val3.txt'
    test_annotations = './UESTC-MMEA-CL/data/test3.txt'
    temporal_slice = 16  # number of sampled frames
    data_format = 'channels_last'  # channels_first or [channels_last]
    resolution = 172

    transform = A.Compose([
        A.Resize(resolution, resolution, always_apply=True),
        A.ToFloat()
    ])

    train_generator = UESTCVideoDataGenerator(
        video_dir=fold,
        annotation_file_path=train_annotations,
        sampler=RandomTemporalSegmentSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        shuffle=True,
    )

    val_generator = UESTCVideoDataGenerator(
        video_dir=fold,
        annotation_file_path=val_annotations,
        sampler=SystematicSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        shuffle=False,
    )

    test_generator = UESTCVideoDataGenerator(
        video_dir=fold,
        annotation_file_path=test_annotations,
        sampler=SystematicSampler(n_frames=temporal_slice),
        to_rgb=True,
        transform=transform,
        use_albumentations=True,
        data_format=data_format,
        shuffle=False,
    )

    batch_size = 16

    # Detect if we have a GPU available
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(val_generator, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': torch.utils.data.DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=4)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_EPOCHS = 15
    model = MoViNet(_C.MODEL.MoViNetA1, causal=False, pretrained=True)
    start_time = time.time()

    trloss_val, tsloss_val = [], []
    model.classifier[3] = torch.nn.Conv3d(2048, 32, (1, 1, 1))
    optimz = optim.Adam(model.parameters(), lr=0.00005)
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, dataloaders_dict['train'], trloss_val)
        evaluate(model, dataloaders_dict['val'], tsloss_val)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')