# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2021/07/28 11:31:29
@Author  :   Liu Qidong
@Version :   2.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import random
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import torch
from torch.utils.data import Dataset, DataLoader


class RecData(Dataset):
    '''Input torch tensor for taking out data.'''
    def __init__(self, inter, user_feature, item_feature, uid, iid, 
                 feat_list, behavior_train, behavior_test, i2index=None, mode='train'):
        super().__init__()
        self.inter = inter
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.uid_dict = uid
        self.iid_dict = iid
        self.behavior_train = behavior_train
        self.behavior_test = behavior_test
        self.i2index = i2index # 记录线下验证集采样后的 index 到原 index
        self.mode = mode
        self.feat_list = feat_list

    def __len__(self):

        return len(self.inter)

    def __getitem__(self, index):
        instance, instance_age, instance_gender, keywords, keywords_p, behavior, behavior_mask = self._merge_features(self.inter[index], index)
        if self.mode == 'train':
            return torch.LongTensor(instance[:-1]), torch.FloatTensor(instance_age), \
                   torch.FloatTensor(instance_gender), torch.FloatTensor(keywords), \
                   torch.FloatTensor(keywords_p), torch.FloatTensor(behavior), \
                   torch.FloatTensor(behavior_mask), torch.FloatTensor([instance[-1]])
        elif self.mode == 'test':
            return torch.LongTensor(instance), torch.FloatTensor(instance_age), \
                   torch.FloatTensor(instance_gender), torch.FloatTensor(keywords), \
                   torch.FloatTensor(keywords_p), torch.FloatTensor(behavior), torch.FloatTensor(behavior_mask)
        else:
            raise ValueError


    def _merge_features(self, inter, index):
        '''merge features into interaction data'''

        # 采样后的验证集, 需要将索引映射为采样前的索引
        if self.i2index is not None:
            index = self.i2index[index]

        user_id = self.uid_dict[inter[0]]
        item_id = self.iid_dict[inter[1]]
        # 取训练集中的其他特征
        label = list(inter[2:])

        # 根据索引取该条交互记录对应的用户历史行为序列
        if self.mode == 'train' and self.i2index is None: # 线上线下训练
            behavior = self.behavior_train[index]
        else:
            behavior = self.behavior_test[index]

        for i in range(len(behavior)):
            behavior[i] = self.iid_dict[behavior[i]] if behavior[i] in self.iid_dict else 0
        
        if behavior[0] == 0:
            behavior[0] = 633391

        behavior = fix_behavior_length(behavior, 10)
        behavior_mask = get_behavior_mask(behavior)

        user_feature = list(self.user_feature[user_id][:-2])
        item_feature = list(self.item_feature[item_id][:-2])
        instance = user_feature + item_feature + label

        # 处理user_age和user_gender
        instance_age = list(self.user_feature[user_id][-2])
        instance_gender = list(self.user_feature[user_id][-1])

        # 处理item_keywords
        keywords = list(self.item_feature[item_id][-2])
        keywords_p = list(self.item_feature[item_id][-1])

        return instance, instance_age, instance_gender, keywords, keywords_p, behavior, behavior_mask


class DataGenerator():
    '''Generate data for model.'''
    def __init__(self, args, feat_list, user_feats=['user_id'], 
                 item_feats=['item_id'], train_feats=[], mode='offline'):

        self.feat_list = feat_list
        self.mode = mode
        self.args = args
        self.data_path = './data/'
        self._load_data()
        self._map_features(user_feats, item_feats)
        self.features = user_feats + item_feats + train_feats
        self.features.remove('user_age')
        self.features.remove('user_gender')
        self.features.remove('keywords')
        self.features.remove('behavior_id')
        
        
    def _load_data(self):
        '''Load data from pickle.'''
        if self.mode == 'offline':
            with open(self.data_path + 'train.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'validation.pkl', 'rb') as f:
                self.test = pickle.load(f)
            with open(self.data_path + 'behavior_train.pkl', 'rb') as f:
                self.behavior_train = pickle.load(f)
            with open(self.data_path + 'behavior_validation.pkl', 'rb') as f:
                self.behavior_test = pickle.load(f)
        elif self.mode == 'online':
            with open(self.data_path + 'train_online.pkl', 'rb') as f:
                self.train = pickle.load(f)
            with open(self.data_path + 'test.pkl', 'rb') as f:
                self.test = pickle.load(f)
            with open(self.data_path + 'behavior_train_online.pkl', 'rb') as f:
                self.behavior_train = pickle.load(f)
            with open(self.data_path + 'behavior_test.pkl', 'rb') as f:
                self.behavior_test = pickle.load(f)
        else:
            raise ValueError
        
        with open(self.data_path + 'user_feature.pkl', 'rb') as f:
            self.user_feature = pickle.load(f)
        with open(self.data_path + 'item_feature.pkl', 'rb') as f:
            self.item_feature = pickle.load(f)
        with open(self.data_path + 'id_dict.pkl', 'rb') as f:
            self.uid_dict, self.iid_dict = pickle.load(f)


    def _map_features(self, user_feats, item_feats):
        '''Get features that will be used in model'''
        #self.user_feature = self.user_feature[user_feats]
        #self.item_feature = self.item_feature[item_feats]
        #self.user_feature = self.user_feature.to_numpy()
        #self.item_feature = self.item_feature.to_numpy()
        if 'keywords' in item_feats:
            # 特征keywords里面存的是list，需要变成40列（序列最长为40）, 用0补全
            # fix_length 变成固定长度
            self.item_feature['keywords'] = self.item_feature['keywords'].apply(lambda x: fix_length(x,40))
            self.item_feature['keywords_p'] = self.item_feature['keywords_p'].apply(lambda x: fix_length(x,40))
            '''
            # 拼起来
            keys = [] # 存keywords的label
            keys_p = [] # 存keywords的概率
            keywords_feat.apply(lambda x: keys.append(x))
            keywords_p.apply(lambda x: keys_p.append(x))
            '''
            item_feats.append('keywords_p')
            self.user_feature = self.user_feature[user_feats]
            self.item_feature = self.item_feature[item_feats]
            self.user_feature = self.user_feature.to_numpy()
            self.item_feature = self.item_feature.to_numpy()
            item_feats.remove('keywords_p')

        else:
            self.user_feature = self.user_feature[user_feats]
            self.item_feature = self.item_feature[item_feats]
            self.user_feature = self.user_feature.to_numpy()
            self.item_feature = self.item_feature.to_numpy()

    
    def _merge_features(self, inter):
        '''merge features into interaction data'''
        print('Merging Features...')
        data = []
        for pair in tqdm(inter):
            user_id = self.uid_dict[pair[0]]
            item_id = self.iid_dict[pair[1]]
            # 取训练集中的其他特征
            label = list(pair[2:])
            user_feature = list(self.user_feature[user_id])
            item_feature = list(self.item_feature[item_id])
            instance = user_feature + item_feature + label
            data.append(instance)

        return data

    
    def make_train_loader(self):
        '''Make train dataloader'''
        print('Make train data...')

        trainset = RecData(self.train, self.user_feature, self.item_feature,
                           self.uid_dict, self.iid_dict, self.feat_list, 
                           self.behavior_train, self.behavior_test)

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
                              self.uid_dict, self.iid_dict, self.feat_list, 
                              self.behavior_train, self.behavior_test, i2index=index)
        elif self.mode == 'online':
            testset = RecData(self.test, self.user_feature, self.item_feature,
                              self.uid_dict, self.iid_dict, self.feat_list, 
                              self.behavior_train, self.behavior_test, mode='test')
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
    
    behavior = list(map(lambda x: x[5], data))
    behavior_mask = list(map(lambda x: x[6], data))

    x = torch.stack(x)  # (bs, feat_num)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i]

    batch_data['user_age'] = torch.stack(x_age)
    batch_data['user_gender'] = torch.stack(x_gender)
    batch_data['keywords'] = torch.stack(keywords)
    batch_data['keywords_p'] = torch.stack(keywords_p)
    batch_data['behavior_id'] = torch.stack(behavior)
    batch_data['behavior_mask'] = torch.stack(behavior_mask)

    if mode == 'offline':
        y = list(map(lambda x: x[-1], data))
        y = torch.FloatTensor(y)
        #y = torch.stack(y)  # (bs, 1)
        y = y.unsqueeze(1)
        return batch_data, y

    elif mode == 'online':
        return batch_data
    

def fix_length(x,k):
    # x 为输入的list，k为要变成的固定长度
    if x == 'nan':
        return np.zeros(40)
    else:
        if type(x[0]) == type('a'):
            # 先归一化处理
            x = list(map(float, x))
            x = list(np.divide(x,sum(x)))
        for i in range(len(x),k):
            x.append(0)
        return x

def fix_behavior_length(x, length):
    if len(x) >= length:
        return x[(-1 * length):]
    for i in range(len(x), length):
        x.append(0)
    return x

def get_behavior_mask(behavior):
    mask = np.where(np.array(behavior) > 0, 1, 0)
    if sum(mask) != 0:
        mask = mask / sum(mask)
    return mask.tolist()