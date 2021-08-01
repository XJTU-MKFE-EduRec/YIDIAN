# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2020/11/19 09:19:26
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import math


def evaluate_auc(true, pred):
    pred = pred.to('cpu').squeeze().detach().numpy()
    true = true.to('cpu').squeeze().detach().numpy()
    return roc_auc_score(true, pred)


class Evaluation():
    
    def __init__(self, user, item, pred, true, topk=10):
        user = user.to('cpu').squeeze().detach().tolist()
        item = item.to('cpu').squeeze().detach().tolist()
        pred = pred.to('cpu').squeeze().detach().tolist()
        true = true.to('cpu').squeeze().detach().tolist()

        self.subjects = pd.DataFrame({'user_id': user, 
                                      'item_id': item, 
                                      'pred': pred, 
                                      'true': true})
        self.user_num = self.subjects['user_id'].nunique()
        self.subjects['rank'] = self.subjects.groupby('user_id')['pred'].rank(method='first', ascending=False)
        self.subjects.sort_values(['user_id', 'rank'], inplace=True)
        self.subjects = self.subjects.loc[self.subjects['rank']<=topk]


    def get_hr(self):
        '''Evaluate for hit rate'''
        hit = self.subjects.loc[self.subjects['true']==1]
        # drop duplicate sample in terms of user
        hit.drop_duplicates(subset='user_id', keep='first', inplace=True)
        hr = hit.shape[0] / self.user_num
        return hr

    def get_ndcg(self):
        '''Evaluate for NDCG'''
        pos = self.subjects.loc[self.subjects['true']==1]
        pos['ndcg'] = pos['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        ndcg = pos['ndcg'].sum() / self.user_num
        return ndcg



class Evaluation_faiss():

    def __init__(self, train_pos, test_pos, item_index, topk):
        
        self.index_df = pd.DataFrame({
            'item': item_index,
            'rank': list(range(1, len(item_index) + 1)) # the rank start from 1
        })
        self.index_df.drop(self.index_df.loc[self.index_df['item'].isin(train_pos)].index, inplace=True)   # filter item in trainset
        self.index_df['rank'] = self.index_df.index + 1
        self.index_df = self.index_df.iloc[:topk]
        self.index_df = self.index_df.loc[self.index_df['item'].isin(test_pos)]
        self.num_test_pos = len(test_pos)

    def get_hr(self):
        '''Evaluate for hit rate'''
        if self.index_df.shape[0] > 0:
            return 1
        else:
            return 0

    def get_ndcg(self):
        '''Evaluate for NDCG'''
        self.index_df['ndcg'] = self.index_df['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        ndcg = self.index_df['ndcg'].sum()
        return ndcg

    def get_recall(self):
        '''Evaluate for Recall'''
        return self.index_df.shape[0] / self.num_test_pos



