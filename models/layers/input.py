# -*- encoding: utf-8 -*-
'''
@File    :   input.py
@Time    :   2020/11/10 09:58:19
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn

'''
@author: Qidong Liu

Build the base input module. Each module only get one feature.
Feature can be categorized into three types: sparse feature, dense feature
and multi-hot feature. 
'''


class sparseFeat():
    '''
    The sparse feature input module. 

    - Init:
        @ feat_name: each feature name
        @ vocabulary_szie: vocaulary size of sparse feature
        @ embedding_dim: the dimension of embedding layer output
    '''
    def __init__(self, feat_name, vocabulary_size, embedding_dim=8, ) -> None:
        super().__init__()
        self.feat_name = feat_name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim


class denseFeat():
    '''
    The dense feature input module

    - Init:
        @ feat_name: each feature name
        @ index: the index of current feature in data(.pkl)
        @ group_name: can be used to group feature
    '''
    def __init__(self, feat_name) -> None:
        super().__init__()
        self.feat_name = feat_name


class sequenceFeat():
    '''
    The sequence feature input module. 

    - Init:
        @ feat_name: each feature name
        @ vocabulary_szie: vocaulary size of sparse feature
        @ index: the index of current feature in data(.pkl)
        @ embedding_dim: the dimension of embedding layer output
        @ group_name: can be used to group feature
    '''
    def __init__(self, feat_name, vocabulary_size, embedding_dim=8, ) -> None:
        super().__init__()
        self.feat_name = feat_name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

