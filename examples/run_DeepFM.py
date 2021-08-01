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
from models.deepfm import DeepFM
from generators.generator import DataGenerator


def run_DeepFM(args):

    with open('./data/info.json', 'r') as f:
        info = json.load(f)

    '''Step 1: Create item feat and user feat and feature list'''
    user_feats = ['user_id', 'user_device', 'user_system', 'user_province', 'user_city']
    item_feats = ['item_id', 'item_picture', 'item_cluster1']
    train_feats = ['network', 'refresh']
    feat_list = []
    for feat in user_feats + item_feats + train_feats:
        feat_list.append(sparseFeat(feat, info['vocabulary_size'][feat], args.em_dim))

    '''Step 2: Data generator'''
    data_generator = DataGenerator(args, user_feats, item_feats, train_feats)

    '''Step 3: construct model and use cuda'''
    model = DeepFM(args, feat_list, data_generator)

    if args.use_cuda:
        model.to('cuda:' + str(args.device_tab))

    '''Step 3: train model or load model'''
    model.fit()










