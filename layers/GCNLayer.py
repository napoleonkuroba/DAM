import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,step_len:int,
                 is_concat: bool = True,
                 share_weights: bool = False):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.gcn=GraphConvolution(in_features,out_features)
        self.in_features=in_features
        self.out_features=out_features
        self.step_len=step_len
    def forward(self,x,adj):
        x = x.reshape(-1,self.step_len,self.in_features)
        output=torch.empty(0).to("cuda")
        for x_ in x:
            output_=self.gcn(x_,adj)
            output=torch.cat((output,output_),dim=0)
        return output.reshape(-1,self.step_len,self.out_features)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'