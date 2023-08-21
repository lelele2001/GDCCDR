from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp



class singleDatasetInfo(object):
    def __init__(self,config,mode):
        self.config = config
        self.mode = mode
        self.root_dir = config['dataset_path']
        if self.mode == 'A':
            self.dataName = config['src']
            self.train_path = self.root_dir + config['src']+'_'+ config['tgt'] +'/train.txt'
            self.test_path = self.root_dir +  config['src']+'_'+ config['tgt'] +'/test.txt'
            self.neg_path = self.root_dir +  config['src']+'_'+ config['tgt'] +'/test_neg_99.txt'
        else:
            self.dataName = config['tgt']
            self.train_path = self.root_dir + config['tgt']+'_'+ config['src'] +'/train.txt'
            self.test_path = self.root_dir + config['tgt']+'_'+ config['src'] +'/test.txt'
            self.neg_path = self.root_dir + config['tgt']+'_'+ config['src'] +'/test_neg_99.txt'
        self.allPos,self.n_user,self.n_item,self.trainlen,self.adj_mat,self.one_adj_mat = self.getTrainDict()
        tmp = self.adj_mat.tocoo()
        self.h_list = list(tmp.row)
        self.t_list = list(tmp.col)
        self.v_list = list(tmp.data)
        tmp = self.one_adj_mat.tocoo()
        self.ui_h_list = list(tmp.row)
        self.ui_t_list = list(tmp.col)
        self.ui_v_list = list(tmp.data)

    def getTrainDict(self):
        train_dict={}
        allPos =[]
        num_user = 0
        num_item = 0
        trainlen = 0
        trainUser, trainItem = [], []
        with open(self.train_path,'r') as f:
            for line in f:
                line = line.strip('\n')
                lines = line.split('\t')
                user = int(lines[0])
                item = int(lines[1])
                trainUser.append(user)
                trainItem.append(item)
                num_user = max(user,num_user)
                num_item = max(item,num_item)
                if user not in train_dict:
                    train_dict[user] = []
                train_dict[user].append(item)
                trainlen += 1
        num_user = num_user + 1
        num_item = num_item + 1

        for user in range(0, num_user):
            posItems = train_dict[user]
            allPos.append(posItems)

        rows = np.concatenate([trainUser, [x + num_user for x in trainItem]], axis=0)
        cols = np.concatenate([[x+num_user for x in trainItem], trainUser], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=[num_user + num_item, num_user + num_item]).tocsr()
        one_adj_mat = sp.coo_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=[num_user, num_item]).tocsr()

        return allPos,num_user,num_item,trainlen,adj_mat,one_adj_mat

def getTrainData(dataset):
    n_user = dataset.n_user
    n_item = dataset.n_item
    trainDataSize = dataset.trainlen
    train_dict = dataset.allPos
    S = sample_TrainData(n_user, n_item,trainDataSize, train_dict)
    return S

def sample_TrainData(n_user, n_item, train_len, train_dict):
    S = []
    perUserNum = train_len//n_user
    for user in range(0,n_user):
        posItems = train_dict[user]
        for _ in  range(0,perUserNum):
            posindex = np.random.randint(0, len(posItems))
            positem = posItems[posindex]
            while True:
                negitem = np.random.randint(0, n_item)
                if negitem in posItems:
                    continue
                else:
                    break
            S.append([user,positem,negitem])

    return np.array(S)


class dualDataSet(Dataset):
    def __init__(self, uID, iID,  rating):
        self.uID = uID
        self.iID = iID
        self.rating = rating
    def __getitem__(self, index):
        return self.uID[index], self.iID[index], self.rating[index]
    def __len__(self):
        return len(self.rating)

class dualDataSetTest(Dataset):
    def __init__(self, uID, iID):
        self.uID = uID
        self.iID = iID
    def __getitem__(self, index):
        return self.uID[index], self.iID[index]
    def __len__(self):
        return len(self.uID)





            


