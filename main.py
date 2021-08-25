# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/11/05 15:38:35
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import json
from models.layers.input import *
from models.deepfm import DeepFM
from models.dnn import DNN
from models.xdeepfm import xDeepFM
from generators.generator import DataGenerator

import os
import time
import argparse
import setproctitle
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Input args')

'''The arguments about model and training'''
parser.add_argument('-m', default='fm', 
                    choices=['dnn', 'deepfm', 'xdeepfm'], help='choose model')
parser.add_argument('-dataset', default='ML1M', 
                    choices=['ML1M', 'Amazon', 'ML20M'], 
                    help='choose dataset')
parser.add_argument('-lr', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('-epoch', default=10, type=int, 
                    help='the number of epoch')
parser.add_argument('-bs', default=128, type=int, help='batch size')
parser.add_argument('-lr-decay', default=0.97, type=float, 
                    help='learning rate decay')
parser.add_argument('-lr-type', default='none', 
                    choices=['exp', 'step', 'cos', 'none'], 
                    help='method of learning rate decay')
parser.add_argument('-period', default=100, type=int, 
                    help='how many epochs to conduct a learning rate decay')
parser.add_argument('-es', default=False, action='store_true', 
                    help='early stop or not')
parser.add_argument('-optimizer', default='adam', 
                    choices=['sgd', 'adam', 'rmsprop'], 
                    help='choose the optimization of training')
parser.add_argument('-bn', default=0, type=bool, 
                    help='batch normalization or not')
parser.add_argument('-init', default='normal', 
                    choices=['normal', 'kaiming'], 
                    help='how initialize the neural network')
parser.add_argument('-load-m', default=False, action='store_true',
                    help='wheather load model rather than training.')
parser.add_argument('-bpr', default=False, action='store_true',
                    help='whether use bpr loss')
parser.add_argument('-online', default=False, action='store_true',
                    help='whether train online')

'''The arguments about log'''
parser.add_argument('-log', default=False, action='store_true', 
                    help='whether log the experiment.')
parser.add_argument('-batch-record', default=1000, type=int, 
                    help='output record once pre <batch-record> batches')


'''The arguments about device'''
parser.add_argument('-num-workers', default=0, type=int, 
                    help='equals to the number of cpu cores')
parser.add_argument('-use-cuda', default=False, action='store_true', 
                    help='whether use cuda')
parser.add_argument('-device-tab', default=0, type=int, 
                    help="specify the device number. note how many gpus you have.")

'''The arguments about specified model'''
parser.add_argument('-em-dim', default=16, type=int,
                    help='the dimension of embedding')





def main(args, mode='offline'):

    with open('./data/info.json', 'r') as f:
        info = json.load(f)

    '''Step 1: Create item feat and user feat and feature list'''
    user_feats = ['user_id', 'user_device', 'user_system', 'user_province', 'user_city', 'user_age', 'user_gender']
    item_feats = ['item_id', 'item_picture', 'item_cluster1', 'item_cluster2', 'keywords']
    train_feats = ['network', 'refresh', 'behavior_id']
    feat_list = []
    for feat in user_feats + item_feats + train_feats:
        if feat in info['feat_type']['sparse']:
            feat_list.append(sparseFeat(feat, info['vocabulary_size'][feat], args.em_dim))
        elif feat in info['feat_type']['multi-hot']:
            feat_list.append(sequenceFeat(feat, info['vocabulary_size'][feat], args.em_dim))
        else:
            raise ValueError
    
    '''Step 2: Data generator'''
    data_generator = DataGenerator(args, feat_list, user_feats, item_feats, train_feats, mode)

    '''Step 3: construct model and use cuda'''
    if args.m == 'dnn':
        Model = DNN
    elif args.m == 'deepfm':
        Model = DeepFM
    elif args.m == 'xdeepfm':
        Model = xDeepFM
    else:
        raise ValueError

    cin_size = [20, 20, 20]
    model = Model(args, cin_size, feat_list, data_generator)

    if args.use_cuda:
        model.to('cuda:' + str(args.device_tab))

    '''Step 3: train model or load model'''
    if mode == 'offline':
        auc = model.fit(mode)
    elif mode == 'online':
        model.fit(mode)
    else:
        raise ValueError

    if mode == 'online':
        '''
        model_path = './save_model/' + args.m + '.ckpt'
        if os.path.exists(model_path):
            model = Model(args, feat_list, data_generator)
            model.load_state_dict(torch.load(model_path))
            if args.use_cuda:
                model.to('cuda:' + str(args.device_tab))
        else:
            model._save_model()
        '''

        with torch.no_grad():
            submit(data_generator, model)
    elif mode == 'offline':
        return auc
    else:
        raise ValueError


    print('Mission Complete!')

    


def submit(DG, model):

    loader = DG.make_test_loader()
    y = model._move_device(torch.Tensor())
    for batch in tqdm(loader):
        x = model._move_device(batch)
        batch_y = model(x)
        y = torch.cat([y, batch_y], dim=0)
    
    y = y.squeeze().cpu().detach().numpy()
    df = pd.DataFrame({'id': list(range(1, 50001)), 'label': y})
    now_str = time.strftime("%m%d%H%M", time.localtime())
    df.to_csv('./submission/' + now_str + '.csv', index=False, header=False)



if __name__ == '__main__':

    setproctitle.setproctitle("cczhao's Competition")
    args = parser.parse_args()

    # 是否线上训练
    if args.online:
        main(args, mode='online')
    else:
        main(args, mode='offline')

