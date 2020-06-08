import torch.nn as nn
import torch
import numpy as np
class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

class BaseLinear(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_hidden_layers=1, use_cuda=False):
        super(BaseLinear, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.use_cuda = use_cuda
        # create layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()

        egde1=nn.Linear(self.num_nodes,self.h_dim)
        self.layers.append(egde1)


        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = nn.Linear(self.h_dim,self.h_dim)
            self.layers.append(h2h)
        # h2o
        egde2=nn.Linear(self.h_dim,self.out_dim)
        self.layers.append(egde2)



    def forward(self,norm):
        norm=self.layers[0](norm)
        for layer in self.layers[1:-1]:
            norm = nn.Sigmoid()(layer(norm))
        norm=self.layers[-1](norm)
        return norm.softmax(dim=1)
