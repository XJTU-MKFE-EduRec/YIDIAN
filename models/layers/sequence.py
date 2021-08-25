# -*- encoding: utf-8 -*-
'''
@File    :   sequence.py
@Time    :   2020/12/22 21:01:11
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class AveragePoolingLayer(nn.Module):
    '''
    Compress each vector of user history into one fixed length vector via
    weighted average pooling. If weights are attention scores, this function
    completes atttion pooling. If weights are none, this function is a
    general average pooling.

    - Input:
        @ x: the vector sequence of items in user behavior. #shape: (bs, max_length, v_dim)
        @ padding: padding matrix for x due to different length of user 
                    behavior. #shape: (bs, sequence num)

    - Output:
        @ res: the aggregated vector of sequence. #shape: (bs, v_dim)

    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, padding, weight=None):
        padding_num = torch.sum(padding, dim=1, keepdim=True)
        padding = padding.unsqueeze(2)
        res = x * padding
        if weight is not None:
            res = weight * res
        res = torch.sum(res, dim=1)
        res = res / padding_num
        return res


class AttentionWeight(nn.Module):
    '''
    A general attention layer for attention weight.
    The attention comes from the following paper:
        ref: Guorui Zhou et al, Deep Interest Network for Click-Through 
        Rate Prediction(2018), KDD'18

    Init:
        @ input_dim: dimension of key and query

    Input:
        @ key: vector of key
        @ query: vector of query

    Output:
        @ w: attention weight for pair (key, query)
    '''
    def __init__(self, input_dim):
        super().__init__()
        self.L1 = nn.Linear(input_dim * 3, 64)
        self.L2 = nn.Linear(64, 1)
        self.act = nn.PReLU()

    def forward(self, key, query):
        out_product = key * query   # corresponding element multiply. #shape: (bs, v_dim)
        con = torch.cat([key, out_product, query], dim=1)   #shape: (bs, 3*v_dim)
        w = self.act(self.L1(con))
        w = self.L2(w)  #shape: (bs,1)
        return w
    


class AttentionWeightLayer(nn.Module):
    '''
    Motivation:
        Only part of user behavior would have effect on the current item.
        So, use attention layer to get weight for each behavior, and get
        various user feature for diffenrent target item.
    
    - Init:
        @ input_dim: dimension of item
    
    - Input:
        @ key_sequence: sequence of key embedding #shape: (bs, max_length, v_dim)
        @ query: query embedding #shape: (bs, v_dim)

    - Output:
        @ res: the weight of each item in sequence via attentionPoolingLayer. #shape: (bs, max_length, 1)
    '''
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.attention_w = AttentionWeight(input_dim)

    def forward(self, key_sequence, query):
        w_list = []
        for i in range(key_sequence.shape[1]):
            key = key_sequence[:, i, :]     # (bs, v_dim)
            w = self.attention_w(key, query)    # (bs, 1)
            w_list.append(w)
        res = torch.cat(w_list, dim=1)  # (bs, max_length)
        res = res.unsqueeze(dim=2)  # (bs, max_length, 1)
        return res








