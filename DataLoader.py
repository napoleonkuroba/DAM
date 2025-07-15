import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def creat_data(dataset, time_steps):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), :]
        b = dataset[i + 1:(i + time_steps) + 1, :]
        X.append(a)
        Y.append(b)
    return np.array(X), np.array(Y)

def splitSet(dataSet,step):
    dataSet=dataSet.cpu().detach()
    tt_1 = dataSet[:, step-1, 0]
    ut_1 = dataSet[:, step-1, 1]
    lms_1 = dataSet[:, step-1, 2]
    mlr_1 = dataSet[:, step-1, 3]
    cml_1 = dataSet[:, step-1, 4]
    tt_2 = dataSet[:, step-1, 5]
    ut_2 = dataSet[:, step-1, 6]
    lms_2 = dataSet[:, step-1, 7]
    mlr_2 = dataSet[:, step-1, 8]
    cml_2 = dataSet[:, step-1, 9]
    tt_3 = dataSet[:, step-1, 10]
    ut_3 = dataSet[:, step-1, 11]
    lms_3 = dataSet[:, step-1, 12]
    mlr_3 = dataSet[:, step-1, 13]
    cml_3 = dataSet[:, step-1, 14]
    return tt_1, tt_2, tt_3, ut_1, ut_2, ut_3, lms_1, lms_2, lms_3, mlr_1, mlr_2, mlr_3, cml_1, cml_2, cml_3

class LinkDataSet(Dataset):
    def __init__(self, flag='train',distance='10',step=12,device='cpu'):
        assert flag in ['train', 'test']
        data_map = {'train': 0, 'test': 1}
        self.set_data = data_map[flag]
        self.distance=distance
        self.step=step
        self.device=device
        self.__read_data__()

    def __read_data__(self):
        x_train_map={}
        x_test_map = {}
        y_train_map = {}
        y_test_map = {}
        for link in ['1','2','3']:
            path = 'data/' + self.distance + '/Link' + str(link) + '.csv'
            datalink = pd.read_csv(path).values
            X, Y = creat_data(datalink, self.step)
            train_size = int(len(X) * 0.80)
            x_train, x_test = X[0:train_size, :, :], X[train_size:len(X), :, :]
            y_train, y_test = Y[0:train_size, :, :], Y[train_size:len(Y), :, :]
            x_train_map[link]=torch.Tensor(x_train)
            x_test_map[link] = torch.Tensor(x_test)
            y_train_map[link] = torch.Tensor(y_train)
            y_test_map[link] = torch.Tensor(y_test)
        self.xTrain=(torch.cat([x_train_map['1'],x_train_map['2'],x_train_map['3']],dim=2)).to(self.device)
        self.yTrain=(torch.cat([y_train_map['1'],y_train_map['2'],y_train_map['3']],dim=2)).to(self.device)
        self.xTest=(torch.cat([x_test_map['1'],x_test_map['2'],x_test_map['3']],dim=2)).to(self.device)
        self.yTest=(torch.cat([y_test_map['1'],y_test_map['2'],y_test_map['3']],dim=2)).to(self.device)

    def __getitem__(self, index):
        if self.set_data == 0:
            return self.xTrain[index],self.yTrain[index]
        return (self.xTest[index]).to(self.device),(self.yTest[index]).to(self.device)

    def __len__(self):
        if self.set_data==0:
            return len(self.yTrain)
        return len(self.yTest)



def test():
    data = LinkDataSet(flag='test',  distance='10', step=12)
    loader = DataLoader(
        data,
        batch_size=25
    )

    for i, (x, y) in enumerate(loader):
        print(i)
        print(x)

