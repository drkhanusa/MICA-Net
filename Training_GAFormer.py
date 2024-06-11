from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import glob
import copy
from algorithm import GramianAngularSumamationField
from model.GAFormer import gaformer
# Creating custom Dataset classes
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target

    def __getitem__(self, index):

        y = self.target[index]
        y = np.array(y, int)
        y = torch.from_numpy(y)


        samples = pd.read_csv(self.data_path[index], header=None).values
        a, b, c, d, e, f = np.array(samples[:,0], float), np.array(samples[:,1], float), np.array(samples[:,2], float), np.array(samples[:,3], float), np.array(samples[:,4], float), np.array(samples[:,5], float)
        x = GramianAngularSumamationField(a, b, c, d, e, f)
        x = x.transpose(2,0,1)
        x = torch.from_numpy(x).float()

        return x, y

    def __len__(self):
        return len(self.data_path)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    print("Start training !")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    # print("outputs: ", outputs.shape)
                    # print("labels ", labels.shape)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # print("inputs size: ", inputs.size(0))
                # print("loss item: ", loss.item())
                running_loss += loss.item() * inputs.size(0)
                # print("running loss: ", running_loss)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    train_fold = './UESTC-MMEA-CL/data/inertial/train'
    val_fold = './UESTC-MMEA-CL/data/inertial/val'
    test_fold = './UESTC-MMEA-CL/data/inertial/test'

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for file_path in glob.glob(os.path.join(train_fold, '*','*.csv')):
      label = int(file_path.split("/")[-2])-1
      train_x.append(file_path)
      train_y.append(label)

    for file_path in glob.glob(os.path.join(val_fold, '*','*.csv')):
      label = int(file_path.split("/")[-2])-1
      val_x.append(file_path)
      val_y.append(label)

    for file_path in glob.glob(os.path.join(test_fold, '*','*.csv')):
      label = int(file_path.split("/")[-2])-1
      test_x.append(file_path)
      test_y.append(label)

    batch_size = 16

    # Detect if we have a GPU available
    dataloaders_dict = {'train':torch.utils.data.DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=4), 'val':torch.utils.data.DataLoader(MyDataset(val_x, val_y), batch_size=batch_size, shuffle=False, num_workers=4), 'test':torch.utils.data.DataLoader(MyDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=4)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = gaformer().to(device)
    # scratch_optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scratch_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scratch_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist = train_model(model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=40)