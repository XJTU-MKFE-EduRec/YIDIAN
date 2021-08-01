# -*- encoding: utf-8 -*-
'''
@File    :   _loss.py
@Time    :   2021/04/26 21:09:36
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR(nn.Module):
    '''
    Bayesian Personalized Ranking Loss
    loss = - (y_pos - y_neg).sigmoid().log().sum()
    '''
    def __init__(self):
        super(BPR, self).__init__()

    def forward(self, y_ui, y_uj, mask=None):
        y_uij = y_ui - y_uj     # shape: (bs, 1)
        y_uij = F.sigmoid(y_uij)
        y_uij = torch.log(y_uij + 1e-5)
        if not mask:
            loss = -y_uij.squeeze(1).sum()
        else:
            loss = -mask*y_uij.squeeze(1).sum()
        return loss


def bpr_loss(y_ui, y_uj, mask=None):
    
    y_uij = y_ui - y_uj     # shape: (bs, 1)
    y_uij = F.sigmoid(y_uij)
    y_uij = torch.log(y_uij + 1e-5)
    if mask is None:
        loss = torch.mean(-y_uij)
    else:
        loss = torch.mean(-(mask*y_uij))
    
    return loss



