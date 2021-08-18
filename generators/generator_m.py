# -*- encoding: utf-8 -*-
'''
@File    :   generator_m.py
@Time    :   2021/08/11 16:14:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from generators.generator import DataGenerator


class RecData(Dataset):
    '''Input torch tensor for taking out data.'''
    def __init__(self, inter, user_feature, item_feature, uid, iid, 
                 feat_list, mode='train'):
        super().__init__()
        self.inter = inter
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.uid_dict = uid
        self.iid_dict = iid
        self.mode = mode
        self.feat_list = feat_list

    def __len__(self):

        return len(self.inter)

    def __getitem__(self, index):

        instance, instance_age, instance_gender, keywords, keywords_p = self._merge_features(self.inter[index])
        if self.mode == 'train':
            return torch.LongTensor(instance[:-2]), torch.FloatTensor(instance_age), torch.FloatTensor(instance_gender), torch.FloatTensor(keywords), torch.FloatTensor(keywords_p), torch.FloatTensor(instance[-2:])
        elif self.mode == 'test':
            return torch.LongTensor(instance), torch.FloatTensor(instance_age), torch.FloatTensor(instance_gender), torch.FloatTensor(keywords), torch.FloatTensor(keywords_p)
        else:
            raise ValueError


    def _merge_features(self, inter):
        '''merge features into interaction data'''

        user_id = self.uid_dict[inter[0]]
        item_id = self.iid_dict[inter[1]]
        # 取训练集中的其他特征
        label = list(inter[2:])
        user_feature = list(self.user_feature[user_id][:-2])
        item_feature = list(self.item_feature[item_id])
        instance = user_feature + item_feature + label

        instance_age = list(self.user_feature[user_id][-2])
        instance_gender = list(self.user_feature[user_id][-1])

        # 处理item_keywords
        keywords = list(self.item_feature[item_id][-2])
        keywords_p = list(self.item_feature[item_id][-1])

        return instance, instance_age, instance_gender, keywords, keywords_p



class MT_DataGenerator(DataGenerator):

    def __init__(self, args, feat_list, user_feats, item_feats, train_feats, mode):
        super().__init__(args, feat_list, user_feats=user_feats, item_feats=item_feats, train_feats=train_feats, mode=mode)


    def _load_data(self):
        '''Load data from pickle.'''
        if self.mode == 'offline':
            with open(self.data_path + 'train_ml.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'validation_ml.pkl', 'rb') as f:
                self.test = pickle.load(f)
        elif self.mode == 'online':
            with open(self.data_path + 'train_online_ml.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'test_m.pkl', 'rb') as f:
                self.test = pickle.load(f)
        else:
            raise ValueError
        
        with open(self.data_path + 'user_feature.pkl', 'rb') as f:
            self.user_feature = pickle.load(f)
        with open(self.data_path + 'item_feature.pkl', 'rb') as f:
            self.item_feature = pickle.load(f)
        with open(self.data_path + 'id_dict.pkl', 'rb') as f:
            self.uid_dict, self.iid_dict = pickle.load(f)


    def make_train_loader(self):
        '''Make train dataloader'''
        print('Make train data...')

        trainset = RecData(self.train, self.user_feature, self.item_feature,
                           self.uid_dict, self.iid_dict, self.feat_list)

        return DataLoader(trainset, 
                          batch_size=self.args.bs,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          drop_last=True,
                          collate_fn=lambda x: collate_point(x, self.features))


    def make_test_loader(self):
        '''Make test dataloader'''
        print('Make evaluation data...')

        if self.mode == 'offline':
            index = random.choices(list(range(len(self.test))), k=50000)
            testset = RecData(self.test[index], self.user_feature, self.item_feature,
                              self.uid_dict, self.iid_dict, self.feat_list)
        elif self.mode == 'online':
            testset = RecData(self.test, self.user_feature, self.item_feature,
                              self.uid_dict, self.iid_dict, self.feat_list, mode='test')
        # set the max batch size
        bs = min(testset.__len__(), 10000)

        return DataLoader(dataset=testset,
                          batch_size=bs,
                          num_workers=self.args.num_workers,
                          collate_fn=lambda x: collate_point(x, self.features, self.mode))


def collate_point(data, features=['user_id', 'item_id'], mode='offline'):
    '''
    Collate samples for point-wise method.
    Input: (bs, 2)-->(x, y)
    '''
    batch_data = {}
    batch_y = {}

    if mode == 'offline':
        x = list(map(lambda x: x[0], data)) # take out features
        x_age = list(map(lambda x: x[1], data))
        x_gender = list(map(lambda x: x[2], data))
        keywords = list(map(lambda x: x[3], data))
        keywords_p = list(map(lambda x: x[4], data))
    elif mode == 'online':
        #x = data
        x = list(map(lambda x: x[0], data)) # take out features
        x_age = list(map(lambda x: x[1], data))
        x_gender = list(map(lambda x: x[2], data))
        keywords = list(map(lambda x: x[3], data))
        keywords_p = list(map(lambda x: x[4], data))
    else:
        raise ValueError

    x = torch.stack(x)  # (bs, feat_num)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i]

    batch_data['user_age'] = torch.stack(x_age)
    batch_data['user_gender'] = torch.stack(x_gender)
    batch_data['keywords'] = torch.stack(keywords)
    batch_data['keywords_p'] = torch.stack(keywords_p)

    if mode == 'offline':
        y = list(map(lambda x: x[5], data))
        y = torch.stack(y)
        batch_y['ctr'] = y[:, 0].unsqueeze(1)
        batch_y['cvr'] = y[:, 1].unsqueeze(1)
        #y = torch.stack(y)  # (bs, 1)
        #y = y.unsqueeze(1)
        return batch_data, batch_y

    elif mode == 'online':
        return batch_data


