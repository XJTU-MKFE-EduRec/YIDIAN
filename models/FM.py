# -*- encoding: utf-8 -*-
'''
@File    :   FM.py
@Time    :   2020/11/05 15:40:01
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from models.basemodel import BaseModel
import torch
import torch.nn as nn


class FM(BaseModel):

    def __init__(self, args, feat_list, data_generator):

        super(FM, self).__init__(args=args, data_generator=data_generator)
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})

        for feat in feat_list:
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            nn.init.normal_(self.FMLinear[feat.feat_name].weight, mean=0.0, std=0.0001)
            nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
        
        self.out = nn.Sigmoid()

    def forward(self, x):

        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for key in x.keys():
            EMlist.append(self.EMdict[key](x[key]))
            fmlinear += self.FMLinear[key](x[key])  # (bs, 1)
        
        
        '''FM'''
        in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        
        y = yFM + fmlinear
        y = self.out(y)

        return y.float()

