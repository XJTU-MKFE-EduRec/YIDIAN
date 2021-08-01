# -*- encoding: utf-8 -*-
'''
@File    :   selection.py
@Time    :   2021/03/18 16:47:25
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

'''This module is to select various element'''

# here put the import lib
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCELoss, MSELoss
from models.layers._loss import *


def select_optimizer(optimize_type='sgd'):
    '''Select optimizer'''
    if optimize_type == 'sgd':
        return SGD
    elif optimize_type == 'adam':
        return Adam
    else:
        raise NotImplementedError('Such optimizer has not been implemented!')


def select_schedule(schedule_type='none'):
    '''Select scheduler'''
    if schedule_type == 'none':
        return None
    elif schedule_type == 'exp':
        return ExponentialLR
    else:
        raise NotImplementedError('Such scheduler has not been implemented!')


def select_loss(loss_type='bce', reduction='mean'):
    '''Select loss function'''
    if loss_type == 'bce':
        return BCELoss(reduction=reduction)
    elif loss_type == 'mse':
        return MSELoss(reduction=reduction)
    elif loss_type == 'bpr':
        return bpr_loss
    else:
        return NotImplementedError('Such loss function has not been implemented!')




