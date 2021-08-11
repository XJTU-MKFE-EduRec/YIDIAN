# -*- encoding: utf-8 -*-
'''
@File    :   dnn.py
@Time    :   2021/08/11 17:32:58
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from collections import OrderedDict
from models.basemodel import BaseModel
from models.layers.input import *
from utils.utils import load_title
import torch
import torch.nn as nn


class DNN(BaseModel):
    def __init__(self, args, feat_list, data_generator):
        super(DNN, self).__init__(args, data_generator=data_generator)
        
        self.feat_list = feat_list
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            input_size += feat.embedding_dim
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
        
        self.out = nn.Sigmoid()


    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for feat in self.feat_list:
            if isinstance(feat, sparseFeat):
                EMlist.append(self.EMdict[feat.feat_name](x[feat.feat_name].long()))
            elif isinstance(feat, sequenceFeat):
                EMlist.append(self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name]))
            else:
                raise ValueError

        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 1)

        y = self.out(yDNN)

        return y.float()