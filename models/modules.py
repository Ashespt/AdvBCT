import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
class MLP(nn.Module):
    def __init__(self,n_inputs=192,n_outputs=1,mlp_depth=5,mlp_width=256,mlp_dropout=0.2):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs,mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mlp_width,mlp_width),
                    nn.BatchNorm1d(mlp_width)
                )
                for _ in range(mlp_depth-2)
            ]
        )
        self.output = nn.Linear(mlp_width,n_outputs)
        self.n_outputs = n_outputs

    def forward(self,x):
        x = self.input(x)
        x = self.dropout(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ElasticBoundary(nn.Module):
    def __init__(self,num_class):
        super(ElasticBoundary, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_class), requires_grad=True)
        self.register_parameter('bias', None)
        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        '''
        x: rmax-rmin,shape:[num_class]
        '''
        x = x*F.sigmoid(self.weight)
        return x

class Transformation(nn.Module):
    def __init__(self,n_inputs=256,K=4,trans_width=16):
        super(Transformation, self).__init__()
        self.rbt = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_inputs,trans_width),
                    nn.BatchNorm1d(trans_width),
                    nn.ReLU(True),
                    nn.Linear(trans_width, trans_width),
                    nn.BatchNorm1d(trans_width),
                    nn.ReLU(True),
                    nn.Linear(trans_width, n_inputs),
                    nn.BatchNorm1d(n_inputs),
                    nn.ReLU(True)
                )
                for _ in range(K)
            ]
        )

    def forward(self,x):
        for block in self.rbt:
            x = block(x)+x
        return x