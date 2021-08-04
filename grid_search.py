# -*- encoding: utf-8 -*-
'''
@File    :   grid_search.py
@Time    :   2021/07/13 21:15:13
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from main import main
import os
import argparse
import setproctitle
import logging

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Input args')

'''The arguments about model and training'''
parser.add_argument('-m', default='fm', 
                    choices=['fm', 'deepfm', 'mf'], help='choose model')
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


if __name__ == '__main__':

    setproctitle.setproctitle("Qidong's Competition")
    args = parser.parse_args()

    log_path = r'./log/grid_search/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    f = open(log_path+args.m+'-bs.txt', 'w+')

    l = []

    best_model = {'auc': 0}
    for bs in [4096, 8192, 16384, 32768]:
        for lr in [0.001, 0.0001]:
            for em in [16]:
                args.bs = bs
                args.lr = lr
                args.em_dim = em
                auc = main(args, 'offline')
                f.writelines(str({'batch_size': bs, 'lr': lr, 'embedding_size': em, 'auc': auc}))
                f.writelines('\n')
                if auc > best_model['auc']:
                    best_model['auc'] = auc
                    best_model['batch_size'] = bs
                    best_model['lr'] = lr
                    best_model['embedding_size'] = em

    f.writelines('The best model is ' + str(best_model))
    f.close()





