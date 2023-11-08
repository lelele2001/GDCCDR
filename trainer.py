import os
import numpy as np
from dataset import getTrainData,dualDataSet,singleDatasetInfo
import torch
from torch.utils.data import Dataset, DataLoader
from evaluate import  HR_NDCG
from tensorboardX import SummaryWriter
import time
from model import GDCCDR

def update_lr(opt,config):
    decay_lr = config['decay_lr']
    for param_group in opt.param_groups:
        param_group['lr'] = param_group['lr'] * decay_lr
    return opt

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


def save_log(config):
    time_path = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    os.makedirs('./logs/' + config['src'] + '_' + config['tgt'] + '/' + str(time_path) + '_log/', exist_ok=True)
    time_path = './logs/'+config['src'] + '_' + config['tgt']+'/'+str(time_path)+'_log'
    with open(time_path+'/setting.txt', 'a') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for key in config.keys():
            f.write(key+":"+str(config[key])+'\n')
        f.writelines('seed:'+str(config['seed'])+'\n')
        f.writelines('------------------- end -------------------'+'\n')
    return time_path

def user_unique_index(users):
    users = users.tolist()
    users_set = set()
    unique_index = []
    idx = 0
    for user in users:
        if user not in users_set:
            users_set.add(user)
            unique_index.append(idx)
            idx += 1
    return torch.tensor(unique_index)


def train(config):

    print("*"*40," Run with following settings 🏃 ","*"*40)
    print(config)
    cprint(f"SEED >> " + str(config['seed']))
    config['src'] = config['dataset'].split('_')[0]
    config['tgt'] = config['dataset'].split('_')[1]
    print(f"Current Datastes: Domain A: \033[0;30;43m{config['src']}\033[0m; Domain B: \033[0;30;43m{config['tgt']}\033[0m;")
    print("*"*115)
    ################ Save logs of this training ################
    time_path = save_log(config)
    ################ Get Dataset Information #########################
    
    datasetA = singleDatasetInfo(config,'A')
    datasetB = singleDatasetInfo(config,'B')
    assert datasetA.n_user==datasetB.n_user
    dataName_A,dataName_B = datasetA.dataName,datasetB.dataName
    cprint(f"Common Users >> " + str(datasetA.n_user))
    cprint(f"Items of Domain A >> " + str(datasetA.n_item))
    cprint(f"Items of Domain B >> " + str(datasetB.n_item))

    ################ Generate Tensorboard #############
    w: SummaryWriter = SummaryWriter(time_path)

    ############### Initialization for Testing ####################
    eva1 = HR_NDCG(config,datasetA,'A')
    eva2 = HR_NDCG(config,datasetB,'B')

    ################## Initialization for Training #####################
    model = GDCCDR(config,datasetA,datasetB).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    hr_best_A, ndcg_best_A, Best_epoch_A = -1, -1, -1
    hr_best_B, ndcg_best_B, Best_epoch_B = -1, -1, -1

    ecl_reg = config['ecl_reg']
    pcl_reg = config['pcl_reg']
    l2_reg = config['l2_reg']

    if config['begin_test'] > 0:
        print("The model will begin testing in Epoch {}".format(config['begin_test']))
    print("Loss = loss_bpr + loss_ecl + loss_pcl  + loss_reg")


    for epoch in range(config['Epoch']):
        t1 = time.time()
        model.train()
        step = 0

        train_A = getTrainData(datasetA)
        train_B = getTrainData(datasetB)
        train_dataset_A = dualDataSet(train_A[:, 0], train_A[:, 1], train_A[:, -1])
        train_iter_A = DataLoader(train_dataset_A, batch_size=config['batchsize'], shuffle=True,num_workers=8)
        train_dataset_B = dualDataSet(train_B[:, 0], train_B[:, 1], train_B[:, -1])
        train_iter_B = DataLoader(train_dataset_B, batch_size=config['batchsize'], shuffle=True,num_workers=8)

        batchs_A = iter(train_iter_A)
        batchs_B = iter(train_iter_B)
        max_step = max(len(train_iter_A),len(train_iter_B))
        aver_loss_bpr_A = 0.
        aver_loss_bpr_B = 0.
        aver_loss_reg_A = 0.
        aver_loss_reg_B = 0.
        aver_loss_ecl_A = 0.
        aver_loss_ecl_B = 0.
        aver_loss_pcl_A = 0.
        aver_loss_pcl_B = 0.
        aver_loss_A = 0.
        aver_loss_B = 0.
        while step < max_step:
            if step < len(train_iter_A):
                optimizer.zero_grad()
                loss = 0
                u, i_pos, i_neg = next(batchs_A)
                u_idx = user_unique_index(u)
                u, i_pos, i_neg,u_idx = u.cuda().long(), i_pos.cuda().long(), i_neg.cuda().long(),u_idx.cuda().long()
                loss_reg,loss_bpr,loss_ecl,loss_pcl = model(u, i_pos,i_neg,u_idx, mode='A')

                loss = l2_reg*loss_reg+loss_bpr+ecl_reg*loss_ecl+pcl_reg*loss_pcl

                loss.backward()
                optimizer.step()
                aver_loss_A += loss
                aver_loss_bpr_A += loss_bpr
                aver_loss_reg_A += l2_reg * loss_reg
                aver_loss_ecl_A += ecl_reg * loss_ecl
                aver_loss_pcl_A += pcl_reg * loss_pcl


            if step < len(train_iter_B):
                optimizer.zero_grad()
                loss = 0
                u, i_pos, i_neg = next(batchs_B)
                u_idx = user_unique_index(u)
                u, i_pos, i_neg,u_idx = u.cuda().long(), i_pos.cuda().long(), i_neg.cuda().long(),u_idx.cuda().long()
                loss_reg,loss_bpr,loss_ecl,loss_pcl = model(u, i_pos, i_neg, u_idx, mode='B')

                loss = l2_reg*loss_reg+loss_bpr+ecl_reg*loss_ecl+pcl_reg*loss_pcl

                loss.backward()
                optimizer.step()
                aver_loss_B += loss
                aver_loss_bpr_B += loss_bpr
                aver_loss_ecl_B += ecl_reg * loss_ecl
                aver_loss_reg_B += l2_reg * loss_reg
                aver_loss_pcl_B += pcl_reg * loss_pcl
            step += 1
        aver_loss_A = aver_loss_A / len(train_iter_A)
        aver_loss_bpr_A = aver_loss_bpr_A / len(train_iter_A)
        aver_loss_reg_A = aver_loss_reg_A / len(train_iter_A)
        aver_loss_ecl_A = aver_loss_ecl_A / len(train_iter_A)
        aver_loss_pcl_A = aver_loss_pcl_A / len(train_iter_A)

        aver_loss_B = aver_loss_B / len(train_iter_B)
        aver_loss_bpr_B = aver_loss_bpr_B / len(train_iter_B)
        aver_loss_reg_B = aver_loss_reg_B / len(train_iter_B)
        aver_loss_ecl_B = aver_loss_ecl_B / len(train_iter_B)
        aver_loss_pcl_B = aver_loss_pcl_B / len(train_iter_B)
        ####################tensorboard################
        w.add_scalar(f'ALoss/loss', aver_loss_A,epoch)
        w.add_scalar(f'ALoss/loss_bpr', aver_loss_bpr_A,epoch)
        w.add_scalar(f'ALoss/loss_reg', aver_loss_reg_A, epoch)
        w.add_scalar(f'ALoss/loss_ecl', aver_loss_ecl_A, epoch)
        w.add_scalar(f'ALoss/loss_pcl', aver_loss_pcl_A, epoch)

        w.add_scalar(f'BLoss/loss', aver_loss_B, epoch)
        w.add_scalar(f'BLoss/loss_bpr', aver_loss_bpr_B, epoch)
        w.add_scalar(f'BLoss/loss_reg', aver_loss_reg_B, epoch)
        w.add_scalar(f'BLoss/loss_ecl', aver_loss_ecl_B, epoch)
        w.add_scalar(f'BLoss/loss_pcl', aver_loss_pcl_B, epoch)
        t2 = time.time()
        print('Train Time: {:.4f}s! Epoch: {}! LossA: {:.6f} = {:.6f}+{:.6f}+{:.6f}+{:.6f}; LossB: {:.6f} = {:.6f}+{:.6f}+{:.6f}+{:.6f}'.format((t2-t1),epoch+1,aver_loss_A,
                aver_loss_bpr_A,aver_loss_ecl_A,aver_loss_pcl_A,aver_loss_reg_A,aver_loss_B,aver_loss_bpr_B,aver_loss_ecl_B,aver_loss_pcl_B,aver_loss_reg_B))
        if epoch<config['begin_test']:
            continue
        ######################## Evaluate #############################
        t1 = time.time()
        hr_A, ndcg_A = eva1.evaluate(model)
        t2 = time.time()
        hr_best_A, ndcg_best_A = max(hr_A, hr_best_A), max(ndcg_A, ndcg_best_A)
        print('Test Time: {:.4f}s; <{: <5}> HR: {:.4f}, NDCG: {:.4f}'.format((t2-t1),dataName_A, hr_A,ndcg_A),end='')
        if hr_A >=hr_best_A:
            Best_epoch_A = epoch+1
            print('\t👍 Best Results! Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(hr_best_A, ndcg_best_A,Best_epoch_A))
        else:
            print('\t   Best Results! Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(hr_best_A, ndcg_best_A,Best_epoch_A))
        w.add_scalar(f'AMetric/HR', hr_A, epoch )
        w.add_scalar(f'AMetric/NDCG', ndcg_A, epoch )
        
        t1 = time.time()
        hr_B, ndcg_B = eva2.evaluate(model)
        t2 = time.time()
        hr_best_B, ndcg_best_B = max(hr_B, hr_best_B), max(ndcg_B, ndcg_best_B)
        print('Test Time: {:.4f}s; <{: <5}> HR: {:.4f}, NDCG: {:.4f}'.format((t2-t1),dataName_B, hr_B,ndcg_B),end='')
        if hr_B >=hr_best_B:
            Best_epoch_B = epoch+1
            print('\t👍 Best Results! Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(hr_best_B, ndcg_best_B,Best_epoch_B))
        else:
            print('\t   Best Results! Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(hr_best_B, ndcg_best_B,Best_epoch_B))
        w.add_scalar(f'BMetric/HR', hr_B, epoch )
        w.add_scalar(f'BMetric/NDCG', ndcg_B, epoch )

        if hr_A<hr_best_A and hr_B < hr_best_B:
            stop_step +=1
        else:
            stop_step = 0
        if stop_step >= config['stop_cnt']:
            print('*'*40,' Early Stop Training At Epoch {} '.format(epoch),'*'*40+'\n')
            with open(time_path + '/setting.txt', 'a') as f:
                f.writelines('------------------ Early Stop Training At Epoch '+str(epoch)+' ------------------' + '\n')
                f.writelines(dataName_A + ": Best_HR: " + str(hr_best_A) + ";  Best_NDCG: " + str(ndcg_best_A) + 'at Epoch '+str(Best_epoch_A)+'\n')
                f.writelines(dataName_B + ": Best_HR: " + str(hr_best_B) + ";  Best_NDCG: " + str(ndcg_best_B) + 'at Epoch '+str(Best_epoch_B)+'\n')
            break
    print('<{}> Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(dataName_A,hr_best_A, ndcg_best_A,Best_epoch_A))
    print('<{}> Best_HR: {:.4f}, Best_NDCG: {:.4f} at Epoch: {}'.format(dataName_B, hr_best_B, ndcg_best_B, Best_epoch_B))
    print('*'*40,' Complete Training Successfully!!! ','*'*40)
    if epoch==config['Epoch']-1:
        with open(time_path + '/setting.txt', 'a') as f:
            f.writelines('------------------ Complete Training Successfully!!! ------------------' + '\n')
            f.writelines(dataName_A + ": Best_HR: " + str(hr_best_A) + ";  Best_NDCG: " + str(ndcg_best_A) + '\n')
            f.writelines(dataName_B + ": Best_HR: " + str(hr_best_B) + ";  Best_NDCG: " + str(ndcg_best_B) + '\n')
    w.close()



