# -*- encoding: utf-8 -*-
'''
@File    :   basemodel.py
@Time    :   2021/07/27 20:19:29
@Author  :   Liu Qidong
@Version :   3.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
import time
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation import evaluate_auc
from utils.selection import *


class BaseModel(nn.Module):
    '''
    The BaseModel of all of models.

    '''
    def __init__(self, args, loss='bce', best_iteration=0, data_generator=None) -> None:
        super().__init__()
        
        self.args = args
        self.loss = loss
        self.best_iteration = best_iteration     # the iteration is used for test 
        self.data_generator = data_generator
        self.metrics = []
        self._init_log()

    
    def fit(self, mode='offline'):

        model = self.train()

        self.logger.info('************** Training **************')
        
        '''Load validation and test data'''
        train_loader = self.data_generator.make_train_loader()
        if mode == 'offline':
            validation_loader = self.data_generator.make_test_loader()
        #test_loader = self.data_generator.make_test_loader()

        optim_ = select_optimizer(self.args.optimizer)
        schedule_ = select_schedule(self.args.lr_type)
        
        self.optimizer = optim_(params=model.parameters(), lr=self.args.lr)
        if schedule_ is not None:
            self.scheduler = schedule_(self.optimizer, 
                                  gamma=self.args.lr_decay, 
                                  last_epoch=-1)
       
        main_metric = []
        for epoch in range(self.args.epoch):

            self.logger.info('************** Load Dataset **************')
            self.logger.info('====Train Epoch: %d/%d====' % (epoch + 1, self.args.epoch))
            
            self.__train_one_epoch__(epoch, train_loader)
            
            # 不要带梯度, 否则不自动清除的话会爆显存
            if mode == 'offline':
                with torch.no_grad():
                    auc = self._evaluate(validation_loader)
                    main_metric.append((auc, epoch))

            if schedule_ is not None:
                self.scheduler.step()
            #TODO:重新实现early stop
        
        if mode == 'offline':
            '''get best iteration according to main metric on validation'''
            main_metric = sorted(main_metric, key=lambda x: x[0], reverse=True)
            self.best_iteration = main_metric[0][1]
            # log the best iteration and the best metrics
            self.logger.info('The best iteration is %d', self.best_iteration)
            self.logger.info('The best result is ' + str(main_metric[self.best_iteration]))


    def __train_one_epoch__(self, epoch, train_loader):

        model = self.train()

        t_loss, t_auc = 0, 0   # set record of loss and auc to 0 only according to i
        i = 0
        '''load dataset for bpr or not'''
    
        train_loss, train_auc = [], []
        '''train part'''
        print('Training...')
        for batch in tqdm(train_loader):
            '''the loss for point-wise and bpr is different'''
            x, y = self._move_device(batch[0]), self._move_device(batch[1])
            y_ = model(x)
            criterion = select_loss(self.loss, reduction='none')
            loss = criterion(y_, y)
            loss = loss.mean()
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t_loss += loss.item() / self.args.batch_record
            t_auc += evaluate_auc(y, y_) / self.args.batch_record

            i += 1
            if not i % self.args.batch_record:
                train_loss.append(t_loss), train_auc.append(t_auc)
                self.writer.add_scalar('train/Loss', t_loss, 
                                            round(i/self.args.batch_record))
                self.writer.add_scalar('train/AUC', t_auc, 
                                            round(i/self.args.batch_record))
                self.logger.info('Epoch %d:(%d) Train Loss: %.5f, Train AUC: %.5f' \
                                     % (epoch+1, i, t_loss, t_auc))
                t_loss, t_auc = 0, 0

    
    def _evaluate(self, loader):
        '''
        Evaluate the model using rank criterion.
        - Input: 
            @ metrics: the list of metrics

        - Output:
            @ res: the dict of metrics
        '''
        model = self.eval()

        self.logger.info('************** Validation Rank Evaluation **************')
        
        y, y_ = self._move_device(torch.Tensor()), self._move_device(torch.Tensor())
        for batch in tqdm(loader):
            x, batch_y = self._move_device(batch[0]), self._move_device(batch[1])
            batch_y_ = model(x)
            y = torch.cat([y, batch_y], dim=0)
            y_ = torch.cat([y_, batch_y_], dim=0)

        auc = evaluate_auc(y, y_)

        self.logger.info('The validation AUC: %.5f' % auc)
        
        return auc

    
    def _initialize_parameters(self, model, init_std=0.0001):
        '''
        Initialize each layer of the model according to various type
        of layer.
        '''
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
            #if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, mean=0, std=0.01)

    
    def _init_log(self):
        '''
        Initialize the logging module. Concretely, initialize the
        tensorboard and logging
        '''
        # judge whether the folder exits
        if not os.path.exists(r'./log/text/' + self.args.m + '/'):
            os.makedirs(r'./log/text/' + self.args.m + '/')

        # get the current time string
        now_str = time.strftime("%m%d%H%M%S", time.localtime())

        '''Initialize tensorboard. Set the save folder.'''
        if self.args.log:
            folder_name = './log/tensorboard/' + self.args.m + '/bs' + str(self.args.bs) + '_lr' + str(self.args.lr) + '_dim' + str(self.args.em_dim) + '/'
        else:
            folder_name = folder_name = './log/tensorboard/' + self.args.m + '/default/'
        self.writer = SummaryWriter(folder_name)

        '''Initialize logging. Create console and file handler'''
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.DEBUG)  # must set
        # create file handler
        if self.args.log:
            log_path = './log/text/'+ self.args.m + '/bs' + str(self.args.bs) + '_lr' + str(self.args.lr) + '_dim' + str(self.args.em_dim) + '.txt'
            self.fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            self.fh.setLevel(logging.DEBUG)
            fm = logging.Formatter("%(asctime)s-%(message)s")
            self.fh.setFormatter(fm)
            self.logger.addHandler(self.fh)

            # record the hyper parameters in the text
            self.logger.info(self.args.m + '\t' + self.args.dataset)
            self.logger.info('learning rate: ' + str(self.args.lr))
            self.logger.info('learning rate decay: ' + str(self.args.lr_decay))
            self.logger.info('batch size: ' + str(self.args.bs))
            self.logger.info('optimizer: ' + str(self.args.optimizer))
            self.logger.info('scheduler: ' + str(self.args.lr_type))
            
        #create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ch)


    def _end_log(self):
        '''
        End the logging module
        '''
        self.writer.close()
        # handler needs to be removed, otherwise, it exists at all
        if self.fh:
            self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)


    def _save_checkpoint(self, epoch):
        '''save checkpoint for each epoch'''
        folder_path = r'./save_model/' + self.args.m + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.state_dict(), folder_path + 'epoch_' + str(epoch) + '.ckpt')

    
    def _load_model(self):
        '''load model at best iteration'''
        check_dir = r'./save_model/' + self.args.m + '/'
        best_path = check_dir + 'epoch_' + str(self.best_iteration) + '.ckpt'
        self.load_state_dict(torch.load(best_path))
        #TODO:删除所有checkpoint,并把最好的模型存储下来

        '''remove all checkpoint files'''
        #for f in os.listdir(check_dir):
        #    os.remove(check_dir + f)

        '''save the best model'''
        torch.save(self.state_dict(), './save_model/' + self.args.m + '.pt')

    
    def _move_device(self, data):
        '''
        move data to specified device.
        '''
        if self.args.use_cuda:
            if isinstance(data, dict):
                for key in data:
                    data[key] = data[key].cuda(self.args.device_tab)
            else:
                data = data.cuda(self.args.device_tab)
            
        return data

    
    def _save_model(self):

        save_dir = r'./save_model/' +self.args.m + '.ckpt'
        torch.save(self.state_dict(), save_dir)

    
    def aggregate_multi_hot(self, EMdict, data):
        '''
        Aggregate multi-hot feature via weighted average.
        
        Input:
        - EMdict: the embedding dict of feature.
        - data: data of the feature. #shape(bs, voca_size)
        
        Output:
        - res: #shape(bs, embedding_dim)
        '''
        index_vector = np.linspace(0, data.shape[1]-1)  # (voca_size)
        index_vector = np.tile(index_vector, (data.shape[0], 1))    # (bs, voca_size)
        index_vector = self._move_device(torch.LongTensor(index_vector))
        index_vector = EMdict(index_vector)
        data = data.unsqueeze(2)
        res = data * index_vector   # (bs, voca_size, embedding_dim)
        res = torch.sum(res, axis=1, keepdim=False) # (bs, embedding_dim)

        return data




