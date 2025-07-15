import numpy as np
import torch.nn as nn
import torch
from layers.AutoCorrelation import AutoCorrelationLayer,AutoCorrelation
from torch.nn import MultiheadAttention
from layers.GCNLayer import Model as GCN

class DAM(nn.Module):
    def __init__(self,model_type):
        super(DAM, self).__init__()
        self.model_type=model_type
        self.f1 = MultiheadAttention(15,3)
        self.a1 = AutoCorrelationLayer(AutoCorrelation(True, 1, attention_dropout=0.05),15, 1)
        self.g1=  GCN(in_features=15,out_features=15,n_heads=4,step_len=3)
        adj = np.ones(9)
        adj_mat = torch.tensor(adj)
        self.adj_mat = adj_mat.reshape(3, 3).to("cuda").float()


    def forward(self, input):
        # 拆分三条链路，每条链路15维特征
        inputs = torch.split(input, 15, dim=2)

        input1 = inputs[0].reshape(-1, 1, 15)
        input2 = inputs[1].reshape(-1, 1, 15)
        input3 = inputs[2].reshape(-1, 1, 15)

        f1, _ = self.f1(input1, input1, input1)
        f2, _ = self.f1(input2, input2, input2)
        f3, _ = self.f1(input3, input3, input3)

        inputa1 = f1.reshape(-1, 6, 15)
        inputa2 = f2.reshape(-1, 6, 15)
        inputa3 = f3.reshape(-1, 6, 15)

        a1, _ = self.a1(inputa1, inputa1, inputa1)
        a2, _ = self.a1(inputa2, inputa2, inputa2)
        a3, _ = self.a1(inputa3, inputa3, inputa3)

        a = torch.cat([a1, a2, a3], dim=2).reshape(-1, 3, 15)
        gout = self.g1(a, self.adj_mat)
        return gout.reshape(-1, 6, 45)

