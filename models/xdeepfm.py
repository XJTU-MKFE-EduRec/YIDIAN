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
from utils.utils import load_title
import torch
import torch.nn as nn


class xDeepFM(BaseModel):
    def __init__(self, args, feat_list, data_generator):
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
        
        self.dnnCTR = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))

        self.dnnCVR = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))
        
        self.outCTR = nn.Sigmoid()
        self.outCVR = nn.Sigmoid()

    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for feat in self.feat_list:
            if isinstance(feat, sparseFeat):
                EMlist.append(self.EMdict[feat.feat_name](x[feat.feat_name].long()))
                fmlinear += self.FMLinear[feat.feat_name](x[feat.feat_name].long())  # (bs, 1)
            elif isinstance(feat, sequenceFeat):
                EMlist.append(self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name]))
                fmlinear += self.aggregate_multi_hot(self.FMLinear[feat.feat_name], x[feat.feat_name])
            else:
                raise ValueError
        
        
        '''CIN'''
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




class CIN(BaseModel):
    '''Compressed Interaction Network(CIN) used in xDeepFM'''
    def __init__(self, args, feat_list, data_generator=None, 
                 sub_module=False):
        super(CIN, self).__init__(args, data_generator=data_generator, 
                                  sub_module=sub_module)
        
        self.feat_list = feat_list





