# -*- encoding: utf-8 -*-
'''
@File    :   run_DeepFM.py
@Time    :   2021/07/29 21:23:42
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import json
from models.layers.input import *
from models.xdeepfm import xDeepFM
from generators.generator import DataGenerator


def run_DeepFM(args):

    with open('./data/info.json', 'r') as f:
        info = json.load(f)

    '''Step 1: Create item feat and user feat and feature list'''
    user_feats = ['user_id', 'user_device', 'user_system', 'user_province', 'user_city', 'user_age', 'user_gender']
    item_feats = ['item_id', 'item_picture', 'item_cluster1']
    train_feats = ['network', 'refresh']
    feat_list = []
    for feat in user_feats + item_feats + train_feats:
        if feat in info['feat_type']['sparse']:
            feat_list.append(sparseFeat(feat, info['vocabulary_size'][feat], args.em_dim))
        elif feat in info['feat_type']['multi-hot']:
            feat_list.append(sequenceFeat(feat, info['vocabulary_size'][feat], args.em_dim))
        else:
            raise ValueError

    '''Step 2: Data generator'''
    data_generator = DataGenerator(args, user_feats, item_feats, train_feats)

    '''Step 3: construct model and use cuda'''
    model = xDeepFM(args, feat_list, data_generator)

    if args.use_cuda:
        model.to('cuda:' + str(args.device_tab))

    '''Step 3: train model or load model'''
    model.fit()










