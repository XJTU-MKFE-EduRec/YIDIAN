# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/08/09 17:26:42
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import torch

def load_title():
    with open(r'./data/title_32.pkl', 'rb') as f:
        title_em = pickle.load(f)

    title_em = torch.tensor(title_em).float()
    return title_em


