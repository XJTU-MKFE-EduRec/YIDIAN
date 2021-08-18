# -*- encoding: utf-8 -*-
'''
@File    :   esmm.py
@Time    :   2021/08/11 15:52:27
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from collections import OrderedDict
from models.basemodel import BaseModel
from models.xdeepfm import CINLayer
from models.layers.input import *
from utils.evaluation import evaluate_auc
from utils.selection import *
import torch
import torch.nn as nn
from tqdm import tqdm


class ESMM(BaseModel):
    def __init__(self, args, cin_size, feat_list, data_generator):
        super(ESMM, self).__init__(args, data_generator=data_generator)
        
        self.feat_list = feat_list
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            input_size += feat.embedding_dim
            nn.init.normal_(self.FMLinear[feat.feat_name].weight, mean=0.0, std=0.0001)
            nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
        
        self.mCTR = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))

        self.mCVR = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))
        
        self.outCTR = nn.Sigmoid()
        self.outCVR = nn.Sigmoid()

        self.cin_list = nn.ModuleList([])
        m = len(feat_list)
        hk = m
        for lsize in cin_size:
            self.cin_list.append(CINLayer(m * hk, lsize))
            hk = lsize
        self.cin_linear = nn.Linear(sum(cin_size), 1)

        self.cin_list_d = nn.ModuleList([])
        m = len(feat_list)
        hk = m
        for lsize in cin_size:
            self.cin_list_d.append(CINLayer(m * hk, lsize))
            hk = lsize
        self.cin_linear_d = nn.Linear(sum(cin_size), 1)


    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for feat in self.feat_list:
            if isinstance(feat, sparseFeat):
                EMlist.append(self.EMdict[feat.feat_name](x[feat.feat_name].long()))
                fmlinear += self.FMLinear[feat.feat_name](x[feat.feat_name].long())  # (bs, 1)
            elif isinstance(feat, sequenceFeat):
                if feat.feat_name == 'keywords':
                    EMlist.append(self.keyword_multi_hot(self.EMdict['keywords'], x['keywords'], x['keywords_p']))
                    fmlinear += self.keyword_multi_hot(self.FMLinear['keywords'], x['keywords'], x['keywords_p'])
                else:
                    EMlist.append(self.aggregate_multi_hot(self.EMdict[feat.feat_name], x[feat.feat_name]))
                    fmlinear += self.aggregate_multi_hot(self.FMLinear[feat.feat_name], x[feat.feat_name])
            else:
                raise ValueError
        
        '''FM'''
        #in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        #square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        #sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        #yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        #yFM += fmlinear
        '''CIN'''
        yCIN = []
        x0 = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        xk = x0
        for cin in self.cin_list:
            cin_res = cin(x0, xk)   # (bs, hk, em_dim)
            xk = cin_res
            yCIN.append(torch.sum(cin_res, dim=2, keepdim=False))    # added vector (bs, hk)
        yCIN = torch.cat(yCIN, dim=1)   # (bs, cin_size)
        yCIN = self.cin_linear(yCIN)

        yCIN_d = []
        x0 = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        xk = x0
        for cin in self.cin_list_d:
            cin_res = cin(x0, xk)   # (bs, hk, em_dim)
            xk = cin_res
            yCIN_d.append(torch.sum(cin_res, dim=2, keepdim=False))    # added vector (bs, hk)
        yCIN_d = torch.cat(yCIN_d, dim=1)   # (bs, cin_size)
        yCIN_d = self.cin_linear(yCIN_d)

        '''CTR model and CVR model'''
        input = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yctr = self.mCTR(input) # (bs, 1)
        ycvr = self.mCVR(input)
        yctr = self.outCTR(yctr+yCIN)
        ycvr += yCIN_d

        yctcvr = yctr * ycvr

        return yctr.float(), yctcvr.float()


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
            yctr_, ycvr_ = model(x)
            criterion_ctr = select_loss('bce')
            criterion_cvr = select_loss('mse')
            loss_ctr = criterion_ctr(yctr_, y['ctr'])
            loss_cvr = criterion_cvr(ycvr_, y['cvr'])
            loss = loss_ctr + self.args.labda * loss_cvr
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t_loss += loss.item() / self.args.batch_record
            t_auc += evaluate_auc(y['ctr'], yctr_) / self.args.batch_record

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


    def _evaluate(self):
        '''
        Evaluate the model using rank criterion.
        - Input: 
            @ metrics: the list of metrics

        - Output:
            @ res: the dict of metrics
        '''
        model = self.eval()

        self.logger.info('************** Validation Rank Evaluation **************')
        
        auc = []
        for i in range(10):
            loader = self.data_generator.make_test_loader()
            y, y_ = self._move_device(torch.Tensor()), self._move_device(torch.Tensor())
            for batch in tqdm(loader):
                x, batch_y = self._move_device(batch[0]), self._move_device(batch[1])
                batch_y_, _ = model(x)
                y = torch.cat([y, batch_y['ctr']], dim=0)
                y_ = torch.cat([y_, batch_y_], dim=0)

            auc.append(evaluate_auc(y, y_))

            self.logger.info('The %dth validation AUC: %.5f' % (i+1, auc[i]))
        
        mean_auc = sum(auc) / len(auc)
        self.logger.info('The mean validation AUC: %.5f' % mean_auc)
        
        return mean_auc



