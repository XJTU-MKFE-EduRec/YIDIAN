# -*- encoding: utf-8 -*-
'''
@File    :   candidate_generator.py
@Time    :   2021/06/11 16:28:14
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import faiss
import numpy as np


class CandidateGenerator():
    def __init__(self, item, args):
        self.item = item
        self.embedding_size = item.shape[1]
        self.make_index(args)

    def make_index(self, args):

        self.make_index_brute_force()

        #if args.use_cuda:

            #res = faiss.StandardGpuResources()
            #self.index = faiss.index_cpu_to_gpu(res, args.device_tab, self.index)

    def make_index_brute_force(self):

        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.item)

    def generate(self, users, k):
        users = np.expand_dims(users, 0)
        # 对users中的每个user产生k个在空间上最相近的物品id
        _, I = self.index.search(users, k)

        return I[0]








