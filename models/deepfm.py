# -*- encoding: utf-8 -*-
'''
@File    :   deepfm.py
@Time    :   2021/05/24 23:05:22
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


class DeepFM(BaseModel):
    def __init__(self, args, feat_list, data_generator):
        super(DeepFM, self).__init__(args, data_generator=data_generator)
        
        self.feat_list = feat_list
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            nn.init.normal_(self.FMLinear[feat.feat_name].weight, mean=0.0, std=0.0001)
            nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
            input_size += feat.embedding_dim
        
        self.dnn = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))
        
        self.out = nn.Sigmoid()

    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for feat in self.feat_list:
            if isinstance(feat, sparseFeat):
                EMlist.append(self.EMdict[feat.feat_name](x[feat.feat_name].long()))
                fmlinear += self.FMLinear[feat.feat_name](x[feat.feat_name].long())  # (bs, 1)
            elif isinstance(feat, sequenceFeat):
                EMlist.append(self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name].long()))
                fmlinear += self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name].long())
            else:
                raise ValueError
        
        
        '''FM'''
        in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        yFM += fmlinear

        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 1)

        y = self.out(yFM + yDNN)

        return y.float()

