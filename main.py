import argparse
import random
import numpy as np
import torch
import os
from trainer import train
import setproctitle
torch.cuda.set_device(0)


def set_seed(seed=2023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_args():
    parser = argparse.ArgumentParser(description='GDCCDR (by SLL)')

    ################# Hyper-parameters of the GDCCDR ####################
    
    parser.add_argument('--ecl_reg', default=0.2, type=float, help='the weight of eliminatory contrastive loss')
    parser.add_argument('--pcl_reg', default=0.001, type=float, help='the weight of proximate contrastive loss')
    parser.add_argument('--alpha', default=0.25, type=float, help='the filter factor')
    parser.add_argument('--beta',  default=0.03, type=float, help='the transfer factor')
    parser.add_argument('--g_layers', default=6, type=int, help='layers of GNN')
    parser.add_argument('--temp', default=0.05, type=float, help='the temperature coefficient of PCL')

    ################# Hyper-parameters of the training ####################
    parser.add_argument('--batchsize', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--Epoch', default=100, type=int)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--topK', default=10, type=int)
    parser.add_argument('--l2_reg', default=0.05, type=float)
    parser.add_argument('--hidden_dim', default=128, help='Embedding dimension', type=int)
    parser.add_argument('--init', default='xavier', type=str)
    parser.add_argument('--stop_cnt', default=8, help='Early Stopping', type=int)
    parser.add_argument('--begin_test', default=0, help='Begin Test At Current Epoch', type=int)
    parser.add_argument('--t_chunk', default=500, help='test users chunks', type=int)
    parser.add_argument('--negNums', default=99, help='Negative Items Numbers', type=int)
    parser.add_argument('--newTest', default=False, help='Whether to generate new negative samples for testing', type=bool)
    parser.add_argument('--dropout', default=0.5, type=float)

    ######################### Dataset choices #######################

    parser.add_argument('--dataset', default='sport_phone', help='sport_phone、elec_phone、sport_cloth、elec_cloth', type=str)
    parser.add_argument('--dataset_path', default='./dataset/',type=str)

    #############################################
    args = parser.parse_args()
    config = {}
    for i in vars(args):
        config[str(i)] = vars(args)[i]
    return config

if __name__ == '__main__':
    setproctitle.setproctitle("GDCCDR")
    config = set_args()
    set_seed(config['seed'])
    ############## Trian Model ###############
    train(config)


