# -*- encoding: utf-8 -*-
'''
@File    :   xdeepfm.py
@Time    :   2021/08/11 14:58:23
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from collections import OrderedDict
from models.basemodel import BaseModel
from models.layers.input import *
import torch
import torch.nn as nn


class xDeepFM(BaseModel):
    def __init__(self, args, cin_size, feat_list, data_generator):
        super(xDeepFM, self).__init__(args, data_generator=data_generator)
        
        self.feat_list = feat_list
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            input_size += feat.embedding_dim
            nn.init.normal_(self.FMLinear[feat.feat_name].weight, mean=0.0, std=0.0001)
            nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
        
        self.dnn = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))

        self.cin_list = nn.ModuleList([])
        m = len(feat_list)
        hk = m
        for lsize in cin_size:
            self.cin_list.append(CINLayer(m * hk, lsize))
            hk = lsize
        self.cin_linear = nn.Linear(sum(cin_size), 1)

        self.out = nn.Sigmoid()


    def forward(self, x):

        EMlist = []
        yLINEAR = 0
        '''get embedding list'''
        for feat in self.feat_list:
            if isinstance(feat, sparseFeat):
                EMlist.append(self.EMdict[feat.feat_name](x[feat.feat_name].long()))
                yLINEAR += self.FMLinear[feat.feat_name](x[feat.feat_name].long())  # (bs, 1)
            elif isinstance(feat, sequenceFeat):
                EMlist.append(self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name]))
                yLINEAR += self.aggregate_multi_hot(self.FMLinear[feat.feat_name], x[feat.feat_name])
            else:
                raise ValueError
        
        '''CIN'''
        yCIN = []
        x0 = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        xk = x0
        for cin in self.cin_list:
            cin_res = cin(x0, xk)   # (bs, hk, em_dim)
            xk = cin_res
            yCIN.append(torch.sum(cin_res, dim=2, keepdim=False))    # added vector (bs, hk)
        yCIN = torch.cat(yCIN, dim=1)   # (bs, cin_size)
        yCIN = self.cin_linear(yCIN)
        
        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 1)

        y = self.out(yCIN + yDNN + yLINEAR)

        return y.float()




class CINLayer(nn.Module):
    '''Compressed Interaction Network(CIN) used in xDeepFM'''
    def __init__(self, in_channels, out_channels):
        '''
        in_channel: m * hk, unfold the feature map to a vector
        out_channel: hk+1, the output size of din
        '''
        super(CINLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)


    def forward(self, x0, xk):
        '''
        x0: shape(bs, m, embedding_dim)
        xk: shape(bs, hk, embedding_dim)
        '''

        # step1: get cubic for outer product
        x0 = x0.permute(0, 2, 1).unsqueeze(3)    # shape(bs, embedding_dim, m, 1)
        xk = xk.permute(0, 2, 1).unsqueeze(2)    # shape(bs, embedding_dim, 1, hk)
        cubic = torch.matmul(x0, xk) # added tensor #shape(bs, embedding_dim, m, hk)

        # step2: complete inetraction via convolution
        cubic = cubic.view(cubic.shape[0], cubic.shape[1], -1)    #shape(bs, embedding_dim, m*hk)
        cubic = cubic.permute(0, 2, 1) #shape(bs, m*hk, embedding_dim)
        res = self.conv(cubic)  # shape(bs, hk+1, embedding_dim)

        return res

