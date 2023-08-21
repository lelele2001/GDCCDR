import numpy as np
from numpy import log2
import torch

class HR_NDCG():
    def __init__(self, config,dataset,mode):
        self.neg_path = dataset.neg_path
        self.test_path = dataset.test_path
        self.allPos = dataset.allPos
        self.n_users = dataset.n_user
        self.n_items = dataset.n_item
        self.dataName = dataset.dataName
    
        self.choose = config['newTest']
        self.negNums = config['negNums']
        self.tck = config['t_chunk']
        if self.tck<=0 or self.tck>=self.n_users:
            self.tck = self.n_users
        if self.choose:
            print('************ Generate New Test Data of {}; *************'.format(self.dataName))
            self.testUsers,self.testItems = self.gen_testData()
            print('************ Finish Generation of {}; *************'.format(self.dataName))
        else:
            self.testUsers,self.testItems = self.read_testData()

        self.mode = mode
        self.topK = config['topK']

    def gen_testData(self):
        test_path = self.test_path
        allPos = self.allPos
        n_items = self.n_items
        testItems = []
        testUsers = []
        allItems = list(range(0, n_items))
        with open(test_path, 'r') as f:
            for line in f:
                if line:
                    line = line.strip('\n')
                    lines = line.split('\t')
                    user = int(lines[0])
                    item = int(lines[1])
                    posItems = allPos[user]
                    negItems = [x for x in allItems if x not in posItems]
                    selNegItems = np.random.choice(negItems, self.negNums, replace=False).tolist()
                    testItems.append(item)
                    testItems.extend(selNegItems)
                    testUsers.extend([user]*(self.negNums + 1))
        return testUsers, testItems

    def read_testData(self):
        neg_path = self.neg_path
        testItems = []
        testUsers = []
        with open(neg_path,'r') as f:
            for line in f:
                if line:
                    line = line.strip('\n')
                    lines = line.split('\t')
                    lines[0] = lines[0].replace("(","").replace(")","")
                    mixs = lines[0].split(',')
                    user = int(mixs[0])
                    raw_item = int(mixs[1])
                    testUsers.append(user)
                    testItems.append(raw_item)
                    for item in lines[1:]:
                        testUsers.append(user)
                        testItems.append(int(item))
        return testUsers,testItems

    def evaluate(self, model):
        EachTestUsers = 10
        EachTestNums = EachTestUsers*(1+self.negNums)
        model = model.eval()
        model.test_Initial(self.mode)
        lenTestUsers = len(self.testUsers)//(1+self.negNums)
        nums = len(self.testUsers)//EachTestNums
        allratings = torch.empty((lenTestUsers,self.negNums+1))
        with torch.no_grad():
            for i in range(0,nums+1):
                s_index = i*EachTestNums
                e_index = min((i+1)*EachTestNums,len(self.testUsers))
                users = torch.LongTensor(self.testUsers[s_index:e_index]).cuda()
                items = torch.LongTensor(self.testItems[s_index:e_index]).cuda()
                preds = model.getRating(users, items).detach().cpu()
                preds = preds.reshape(-1,(1+self.negNums))
                allratings[s_index//(1+self.negNums):e_index//(1+self.negNums)] = preds
            _, rating_K = torch.topk(allratings, k=self.topK)
            rating_K = rating_K.cpu().numpy()
            target = 0

            ############### Parallel Computing HR #######################
            sum_HR = np.sum(rating_K == target)
            HR = sum_HR/lenTestUsers

            ############### Parallel Computing NDCG #######################
            index_NDCG = np.where(rating_K == target)
            tmp_NDCG = index_NDCG[1]
            NDCGs = log2(2) / log2(tmp_NDCG + 2)
            NDCG = np.sum(NDCGs)/lenTestUsers

            return HR,NDCG


