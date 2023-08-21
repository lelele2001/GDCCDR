import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)

class MLP(nn.Module):
    def __init__(self, dim1, dim2, dim3,dim4, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim3)
        self.linear3 = nn.Linear(dim3, dim4,bias=True)
        self.prelu=nn.PReLU()
        self.tanh = nn.Tanh()
    def forward(self, data):
        x = data
        x = self.linear1(x)
        x = self.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        x = self.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.linear3(x)
        x = F.normalize(x,p=2,dim=-1)
        return x
class MetaInvFuse(nn.Module):
    def __init__(self,n_user,n_item,indices_u,indices_i,shape,config):
        super(MetaInvFuse, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.emb_dim = config['hidden_dim']
        self.indices_u = indices_u
        self.indices_i = indices_i
        self.shape = shape
        self.k = 10
        self.values_u = torch.ones(size=(len(self.indices_u[0]), 1)).view(-1).cuda()
        self.values_i = torch.ones(size=(len(self.indices_i[0]), 1)).view(-1).cuda()
        self.meta_netu = nn.Linear(self.emb_dim*4, self.emb_dim, bias=True)
        self.meta_neti = nn.Linear(self.emb_dim*2, self.emb_dim, bias=True)
        self.mlp1 = MLP(self.emb_dim,self.emb_dim*self.k,self.emb_dim//2,self.emb_dim*self.k)
        self.mlp2 = MLP(self.emb_dim,self.emb_dim*self.k,self.emb_dim//2,self.emb_dim*self.k)


    def forward(self,uemb_s,iemb_s,uemb_t,u_emb_sp):
        neighbors_u = torch_sparse.spmm(self.indices_u, self.values_u, self.shape[0], self.shape[1], iemb_s)
        tmp_embu = (self.meta_netu(torch.concat((uemb_s,uemb_t,u_emb_sp,neighbors_u),dim=1).detach()))

        neighbors_i = torch_sparse.spmm(self.indices_i, self.values_i, self.shape[1], self.shape[0], (uemb_s+u_emb_sp)/2)
        tmp_embi = (self.meta_neti(torch.concat((iemb_s,neighbors_i),dim=1).detach()))

        metau=self.mlp1(tmp_embu). reshape(-1,self.emb_dim,self.k)# d*k
        metai=self.mlp2(tmp_embi). reshape(-1,self.k,self.emb_dim)# k*d
        meta_biasu =(torch.mean( metau,dim=0))
        meta_biasi=(torch.mean( metai,dim=0))

        low_weightu=F.softmax( metau + meta_biasu , dim=1)
        low_weighti=F.softmax( metai + meta_biasi ,dim=1)

        return low_weightu,low_weighti

def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))
    return 0.5*loss

class SUCCDR(nn.Module):
    def __init__(self, config,datasetA,datasetB):
        super(SUCCDR, self).__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.config = config
        self.n_userA,self.n_itemA = datasetA.n_user,datasetA.n_item
        self.n_userB,self.n_itemB = datasetB.n_user,datasetB.n_item

        self.shape_A = datasetA.adj_mat.tocoo().shape
        self.shape_B = datasetB.adj_mat.tocoo().shape
        self.ui_shape_A = datasetA.one_adj_mat.tocoo().shape
        self.ui_shape_B = datasetB.one_adj_mat.tocoo().shape


        self.h_list_A = datasetA.h_list
        self.t_list_A = datasetA.t_list
        self.h_list_B = datasetB.h_list
        self.t_list_B = datasetB.t_list

        self.ui_h_list_A = datasetA.ui_h_list
        self.ui_t_list_A = datasetA.ui_t_list
        self.ui_h_list_B = datasetB.ui_h_list
        self.ui_t_list_B = datasetB.ui_t_list

        self.ui_indices_A = torch.tensor([self.ui_h_list_A, self.ui_t_list_A], dtype=torch.long).cuda()
        self.ui_indices_B = torch.tensor([self.ui_h_list_B, self.ui_t_list_B], dtype=torch.long).cuda()
        self.iu_indices_A = torch.tensor([self.ui_t_list_A, self.ui_h_list_A], dtype=torch.long).cuda()
        self.iu_indices_B = torch.tensor([self.ui_t_list_B, self.ui_h_list_B], dtype=torch.long).cuda()
        self.meta_A = MetaInvFuse(self.n_userA,self.n_itemA,self.ui_indices_A,self.iu_indices_A,self.ui_shape_A,config)
        self.meta_B = MetaInvFuse(self.n_userB,self.n_itemB,self.ui_indices_B,self.iu_indices_B,self.ui_shape_B,config)


        self.all_indices_A = torch.tensor([self.h_list_A, self.t_list_A], dtype=torch.long).cuda()
        self.D_indices_A = torch.tensor([list(range(self.n_userA + self.n_itemA)), list(range(self.n_userA + self.n_itemA))], dtype=torch.long).cuda()
        self.all_indices_B = torch.tensor([self.h_list_B, self.t_list_B], dtype=torch.long).cuda()
        self.D_indices_B = torch.tensor([list(range(self.n_userB + self.n_itemB)), list(range(self.n_userB + self.n_itemB))], dtype=torch.long).cuda()
        self.h_list_A = torch.LongTensor(self.h_list_A).cuda()
        self.t_list_A = torch.LongTensor(self.t_list_A).cuda()
        self.h_list_B = torch.LongTensor(self.h_list_B).cuda()
        self.t_list_B = torch.LongTensor(self.t_list_B).cuda()

        self.G_indices_A, self.G_values_A = self._cal_sparse_adj_A()
        self.G_indices_B, self.G_values_B = self._cal_sparse_adj_B()
        self.G_values_inv_A = self.G_values_A
        self.G_values_pri_A = self.G_values_A
        self.G_values_inv_B = self.G_values_B
        self.G_values_pri_B = self.G_values_B

        self.emb_dim = config['hidden_dim']
        self.n_layers = config['g_layers']
        self.init = config['init']
        self.temp = config['temp']
        self.dropout = config['dropout']

        self.init_weight()
        self.act = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.alpha = config['alpha']
        self.beta = config['beta']
        
    def _cal_sparse_adj_A(self):

        A_values = torch.ones(size=(len(self.h_list_A), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.h_list_A, col=self.t_list_A, value=A_values, sparse_sizes=self.shape_A).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices_A, D_values, self.all_indices_A, A_values, self.shape_A[0], self.shape_A[1], self.shape_A[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices_A, D_values, self.shape_A[0], self.shape_A[1], self.shape_A[1])
        return G_indices, G_values

    def _cal_sparse_adj_B(self):

        A_values = torch.ones(size=(len(self.h_list_B), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.h_list_B, col=self.t_list_B, value=A_values, sparse_sizes=self.shape_B).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices_B, D_values, self.all_indices_B, A_values, self.shape_B[0], self.shape_B[1], self.shape_B[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices_B, D_values, self.shape_B[0], self.shape_B[1], self.shape_B[1])
        return G_indices, G_values

    def init_weight(self):

        self.embedding_item_A = torch.nn.Embedding(self.n_itemA, self.emb_dim)
        self.embedding_user_A = torch.nn.Embedding(self.n_userA, self.emb_dim)
        self.embedding_item_B = torch.nn.Embedding(self.n_itemB, self.emb_dim)
        self.embedding_user_B = torch.nn.Embedding(self.n_userB, self.emb_dim)

        self.b_inv_A=nn.Parameter(torch.FloatTensor(1,self.emb_dim))
        self.w_inv_A=nn.Parameter(torch.FloatTensor(self.emb_dim,self.emb_dim))
        self.b_pri_A=nn.Parameter(torch.FloatTensor(1,self.emb_dim))
        self.w_pri_A=nn.Parameter(torch.FloatTensor(self.emb_dim,self.emb_dim))

        self.b_inv_B=nn.Parameter(torch.FloatTensor(1,self.emb_dim))
        self.w_inv_B=nn.Parameter(torch.FloatTensor(self.emb_dim,self.emb_dim))
        self.b_pri_B=nn.Parameter(torch.FloatTensor(1,self.emb_dim))
        self.w_pri_B=nn.Parameter(torch.FloatTensor(self.emb_dim,self.emb_dim))

        if self.init == 'xavier':
            nn.init.xavier_uniform_(self.embedding_item_A.weight)
            nn.init.xavier_uniform_(self.embedding_user_A.weight)
            nn.init.xavier_uniform_(self.embedding_item_B.weight)
            nn.init.xavier_uniform_(self.embedding_user_B.weight)

            nn.init.xavier_uniform_(self.b_inv_A.data)
            nn.init.xavier_uniform_(self.w_inv_A.data)
            nn.init.xavier_uniform_(self.b_pri_A.data)
            nn.init.xavier_uniform_(self.w_pri_A.data)
            nn.init.xavier_uniform_(self.b_inv_B.data)
            nn.init.xavier_uniform_(self.w_inv_B.data)
            nn.init.xavier_uniform_(self.b_pri_B.data)
            nn.init.xavier_uniform_(self.w_pri_B.data)
            print('use xavier initilizer')

        elif self.init == 'normal':
            nn.init.normal_(self.embedding_item_A.weight, std=0.1)
            nn.init.normal_(self.embedding_user_A.weight, std=0.1)
            nn.init.normal_(self.embedding_item_B.weight, std=0.1)
            nn.init.normal_(self.embedding_user_B.weight, std=0.1)

            nn.init.normal_(self.b_inv_A.data, std=0.1)
            nn.init.normal_(self.w_inv_A.data, std=0.1)
            nn.init.normal_(self.b_pri_A.data, std=0.1)
            nn.init.normal_(self.w_pri_A.data, std=0.1)
            nn.init.normal_(self.b_inv_B.data, std=0.1)
            nn.init.normal_(self.w_inv_B.data, std=0.1)
            nn.init.normal_(self.b_pri_B.data, std=0.1)
            nn.init.normal_(self.w_pri_B.data, std=0.1)
            print('use normal initilizer')

        elif self.init == 'pretrain':
            pass


    def filter_graph(self,inv_head_emb,inv_tail_emb,pri_head_emb,pri_tail_emb):

        scores_inv = self.act(torch.sum(inv_head_emb * inv_tail_emb, dim=1)).squeeze()
        scores_pri = self.act(torch.sum(pri_head_emb * pri_tail_emb, dim=1)).squeeze()

        norm_scores_inv, norm_scores_pri = scores_inv,scores_pri
        return norm_scores_inv,norm_scores_pri


    def inference(self):
        embedding_user_pri_A = torch.multiply(self.embedding_user_A.weight, torch.sigmoid(torch.matmul(self.embedding_user_A.weight,self.w_pri_A) + self.b_pri_A))
        embedding_user_inv_A = torch.multiply(self.embedding_user_A.weight, torch.sigmoid(torch.matmul(self.embedding_user_A.weight,self.w_inv_A) + self.b_inv_A))
        embedding_user_pri_B = torch.multiply(self.embedding_user_B.weight, torch.sigmoid(torch.matmul(self.embedding_user_B.weight,self.w_pri_B) + self.b_pri_B))
        embedding_user_inv_B = torch.multiply(self.embedding_user_B.weight, torch.sigmoid(torch.matmul(self.embedding_user_B.weight,self.w_inv_B) + self.b_inv_B))

        embeddings_pri_A = [torch.concat([embedding_user_pri_A, self.embedding_item_A.weight], dim=0)]
        embeddings_inv_A = [torch.concat([embedding_user_inv_A, self.embedding_item_A.weight], dim=0)]
        embeddings_pri_B = [torch.concat([embedding_user_pri_B, self.embedding_item_B.weight], dim=0)]
        embeddings_inv_B = [torch.concat([embedding_user_inv_B, self.embedding_item_B.weight], dim=0)]
        
        for i in range(0, self.n_layers):
            # Graph-based Message Passing
            with torch.no_grad():
                inv_head_embeddings = torch.index_select(embeddings_inv_A[i], 0, self.h_list_A)
                inv_tail_embeddings = torch.index_select(embeddings_inv_A[i], 0, self.t_list_A)
                pri_head_embeddings = torch.index_select(embeddings_pri_A[i], 0, self.h_list_A)
                pri_tail_embeddings = torch.index_select(embeddings_pri_A[i], 0, self.t_list_A)
            F_values_inv_A,F_values_pri_A = self.filter_graph(inv_head_embeddings,inv_tail_embeddings,pri_head_embeddings,pri_tail_embeddings)
            
            filter_embeddings_inv_A = torch_sparse.spmm(self.G_indices_A, F_values_inv_A*self.G_values_A, self.shape_A[0], self.shape_A[1], embeddings_inv_A[i])
            filter_embeddings_pri_A = torch_sparse.spmm(self.G_indices_A, F_values_pri_A*self.G_values_A, self.shape_A[0], self.shape_A[1], embeddings_pri_A[i])
            raw_embeddings_inv_A = torch_sparse.spmm(self.G_indices_A, self.G_values_A, self.shape_A[0], self.shape_A[1], embeddings_inv_A[i])
            raw_embeddings_pri_A = torch_sparse.spmm(self.G_indices_A, self.G_values_A, self.shape_A[0], self.shape_A[1], embeddings_pri_A[i])

            filter_embeddings_inv_A = F.dropout(filter_embeddings_inv_A, self.dropout, training=self.training)
            filter_embeddings_pri_A = F.dropout(filter_embeddings_pri_A, self.dropout, training=self.training)
            raw_embeddings_inv_A = F.dropout(raw_embeddings_inv_A, self.dropout, training=self.training)
            raw_embeddings_pri_A = F.dropout(raw_embeddings_pri_A, self.dropout, training=self.training)
            with torch.no_grad():
                inv_head_embeddings = torch.index_select(embeddings_inv_B[i], 0, self.h_list_B)
                inv_tail_embeddings = torch.index_select(embeddings_inv_B[i], 0, self.t_list_B)
                pri_head_embeddings = torch.index_select(embeddings_pri_B[i], 0, self.h_list_B)
                pri_tail_embeddings = torch.index_select(embeddings_pri_B[i], 0, self.t_list_B)
            F_values_inv_B,F_values_pri_B = self.filter_graph(inv_head_embeddings,inv_tail_embeddings,pri_head_embeddings,pri_tail_embeddings)

            filter_embeddings_inv_B = torch_sparse.spmm(self.G_indices_B, F_values_inv_B*self.G_values_B, self.shape_B[0], self.shape_B[1], embeddings_inv_B[i])
            filter_embeddings_pri_B = torch_sparse.spmm(self.G_indices_B, F_values_pri_B*self.G_values_B, self.shape_B[0], self.shape_B[1], embeddings_pri_B[i])
            raw_embeddings_inv_B = torch_sparse.spmm(self.G_indices_B, self.G_values_B, self.shape_B[0], self.shape_B[1], embeddings_inv_B[i])
            raw_embeddings_pri_B = torch_sparse.spmm(self.G_indices_B, self.G_values_B, self.shape_B[0], self.shape_B[1], embeddings_pri_B[i])

            filter_embeddings_inv_B = F.dropout(filter_embeddings_inv_B, self.dropout, training=self.training)
            filter_embeddings_pri_B = F.dropout(filter_embeddings_pri_B, self.dropout, training=self.training)
            raw_embeddings_inv_B = F.dropout(raw_embeddings_inv_B, self.dropout, training=self.training)
            raw_embeddings_pri_B = F.dropout(raw_embeddings_pri_B, self.dropout, training=self.training)
            
            layer_embeddings_inv_A = self.alpha*filter_embeddings_inv_A + raw_embeddings_inv_A
            layer_embeddings_pri_A = self.alpha*filter_embeddings_pri_A + raw_embeddings_pri_A
            embeddings_inv_A.append(layer_embeddings_inv_A)
            embeddings_pri_A.append(layer_embeddings_pri_A)
            
            layer_embeddings_inv_B = self.alpha*filter_embeddings_inv_B + raw_embeddings_inv_B
            layer_embeddings_pri_B = self.alpha*filter_embeddings_pri_B + raw_embeddings_pri_B
            embeddings_inv_B.append(layer_embeddings_inv_B)
            embeddings_pri_B.append(layer_embeddings_pri_B)


        all_embeddings_pri_A = torch.stack(embeddings_pri_A, dim=1)
        u_embedding_pri_list_A, i_embedding_pri_list_A = torch.split(all_embeddings_pri_A, [self.n_userA, self.n_itemA], 0)

        all_embeddings_inv_A = torch.stack(embeddings_inv_A, dim=1)
        u_embedding_inv_list_A, i_embedding_inv_list_A = torch.split(all_embeddings_inv_A, [self.n_userA, self.n_itemA], 0)

        u_embedding_pri_mean_A = torch.mean(u_embedding_pri_list_A, dim=1, keepdim=False)
        u_embedding_inv_mean_A = torch.mean(u_embedding_inv_list_A, dim=1, keepdim=False)
        i_embedding_A = torch.mean(torch.concat([i_embedding_pri_list_A,i_embedding_inv_list_A[:,1:,:]],1), dim=1, keepdim=False)

        all_embeddings_pri_B = torch.stack(embeddings_pri_B, dim=1)
        u_embedding_pri_list_B, i_embedding_pri_list_B = torch.split(all_embeddings_pri_B, [self.n_userB, self.n_itemB], 0)

        all_embeddings_inv_B = torch.stack(embeddings_inv_B, dim=1)
        u_embedding_inv_list_B, i_embedding_inv_list_B = torch.split(all_embeddings_inv_B, [self.n_userB, self.n_itemB], 0)

        u_embedding_pri_mean_B = torch.mean(u_embedding_pri_list_B, dim=1, keepdim=False)
        u_embedding_inv_mean_B = torch.mean(u_embedding_inv_list_B, dim=1, keepdim=False)
        i_embedding_B = torch.mean(torch.concat([i_embedding_pri_list_B,i_embedding_inv_list_B[:,1:,:]],1), dim=1, keepdim=False)
        return u_embedding_pri_mean_A,u_embedding_inv_mean_A,i_embedding_A,u_embedding_pri_mean_B,u_embedding_inv_mean_B,i_embedding_B,u_embedding_pri_list_A,u_embedding_pri_list_B


    def ssl_loss(self, data1, data2, u_idx):

        embeddings1 = F.embedding(u_idx,data1)
        embeddings2 = F.embedding(u_idx,data2)
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)

        pos_score  = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = torch.exp(pos_score / self.temp)
        all_score  = torch.sum(torch.exp(all_score / self.temp), dim = 1)
        ssl_loss  = -torch.sum(torch.log(pos_score / all_score)) / len(u_idx)
        return ssl_loss
    
    def bpr_loss(self, users_emb_pri, users_inv_pos, users_inv_neg, pos_emb,neg_emb):

        pos_scores_pri = torch.sum(torch.mul(users_emb_pri, pos_emb), dim=1)
        pos_scores_inv = torch.sum(torch.mul(users_inv_pos, pos_emb), dim=1)

        neg_scores_pri = torch.sum(torch.mul(users_emb_pri, neg_emb), dim=1)
        neg_scores_inv = torch.sum(torch.mul(users_inv_neg, neg_emb), dim=1)

        loss_bpr_all = torch.mean(nn.functional.softplus((neg_scores_pri + neg_scores_inv) - (pos_scores_pri + pos_scores_inv)))
        return loss_bpr_all
    
    def ecl_loss(self, users_self, users_cro, items_emb):

        pos_score = torch.sum(torch.mul(users_self, items_emb), dim=1)
        pos_score = torch.exp(pos_score)
        norm_item_add = items_emb.unsqueeze(1)
        all_score = torch.sum(torch.mul(users_cro, norm_item_add), dim=2)
        all_score = torch.sum(torch.exp(all_score), dim=1)
        ecl_loss = -torch.sum(torch.log(pos_score / (all_score + pos_score))) / len(users_self)
        return ecl_loss
    
    def Ptransfer(self,trans1,trans2,u_self,u_cro):
        temp = torch.sum(torch.multiply(u_cro.unsqueeze(-1),trans1), dim=1)
        tran_uemb = torch.sum(torch.multiply(temp.unsqueeze(-1),trans2), dim=1)
        fused_inv =  u_self + self.beta * (tran_uemb + u_cro)
        return fused_inv, tran_uemb + u_cro
    
    def forward(self, users, items_pos,items_neg,u_idx = None,mode='A'):
        u_pri_A,u_inv_A,i_A,u_pri_B,u_inv_B,i_B,u_pri_list_A,u_pri_list_B = self.inference()

        u_emb_pri_A = u_pri_A[users]
        u_emb_inv_A = u_inv_A[users]
        u_emb_pri_B = u_pri_B[users]
        u_emb_inv_B = u_inv_B[users]

        if mode=="A":
            pos_emb = i_A[items_pos]
            neg_emb = i_A[items_neg]
            
            trans_u,trans_i = self.meta_A(u_inv_A,i_A,u_inv_B,u_pri_A)
            fused_emb_inv_pos,trans_emb_inv_pos = self.Ptransfer(trans_u[users],trans_i[items_pos],u_emb_inv_A,u_emb_inv_B)
            fused_emb_inv_neg,trans_emb_inv_neg = self.Ptransfer(trans_u[users],trans_i[items_neg],u_emb_inv_A,u_emb_inv_B)

            loss_bpr_all = self.bpr_loss(u_emb_pri_A,fused_emb_inv_pos, fused_emb_inv_neg, pos_emb, neg_emb)

            cro_u_emb_pri = u_pri_list_B[users]
            loss_ecl = self.ecl_loss(u_emb_pri_A,cro_u_emb_pri,pos_emb)

            loss_reg = (1 / 2) * (torch.norm(self.embedding_user_A(users.long()), p=2).pow(2) +
                                torch.norm(self.embedding_item_A(items_pos.long()), p=2).pow(2) +
                                torch.norm(self.embedding_item_A(items_neg.long()), p=2).pow(2)) / float(len(users))

            loss_ssl = self.ssl_loss(u_emb_inv_A,trans_emb_inv_pos,u_idx)

            return loss_reg,loss_bpr_all,loss_ecl,loss_ssl

        if mode=="B":

            pos_emb = i_B[items_pos]
            neg_emb = i_B[items_neg]

            trans_u,trans_i = self.meta_B(u_inv_B,i_B,u_inv_A,u_pri_B)
            fused_emb_inv_pos,trans_emb_inv_pos = self.Ptransfer(trans_u[users],trans_i[items_pos],u_emb_inv_B,u_emb_inv_A)
            fused_emb_inv_neg,trans_emb_inv_neg = self.Ptransfer(trans_u[users],trans_i[items_neg],u_emb_inv_B,u_emb_inv_A)

            loss_bpr_all = self.bpr_loss(u_emb_pri_B,fused_emb_inv_pos,fused_emb_inv_neg, pos_emb, neg_emb)

            cro_u_emb_pri = u_pri_list_A[users]
            loss_ecl = self.ecl_loss(u_emb_pri_B,cro_u_emb_pri,pos_emb)

            loss_reg = (1 / 2) * (torch.norm(self.embedding_user_B(users.long()), p=2).pow(2) +
                                torch.norm(self.embedding_item_B(items_pos.long()), p=2).pow(2) +
                                torch.norm(self.embedding_item_B(items_neg.long()), p=2).pow(2)) / float(len(users))


            loss_ssl = self.ssl_loss(u_emb_inv_B,trans_emb_inv_pos,u_idx)

            return loss_reg,loss_bpr_all,loss_ecl,loss_ssl
    def test_Initial(self,mode):
        u_pri_A,u_inv_A,i_A,u_pri_B,u_inv_B,i_B,_,_ = self.inference()
        
        if mode == "A":
            self.u_pri = u_pri_A
            self.trans_u,self.trans_i = self.meta_A(u_inv_A, i_A, u_inv_B,u_pri_A)
            self.i_all = i_A
            self.inv_self,self.inv_cro = u_inv_A,u_inv_B
        else:
            self.u_pri = u_pri_B
            self.trans_u,self.trans_i = self.meta_B(u_inv_B,i_B,u_inv_A,u_pri_B)
            self.i_all= i_B
            self.inv_self,self.inv_cro = u_inv_B,u_inv_A

    def getRating(self, users, items):
            
        u_emb_inv,_= self.Ptransfer(self.trans_u[users],self.trans_i[items],self.inv_self[users],self.inv_cro[users])
        
        users_emb_pri = self.u_pri[users]
        users_emb_inv = u_emb_inv
        items_emb = self.i_all[items]

        scores_pri = torch.sum(torch.multiply(users_emb_pri, items_emb),dim=1)
        scores_inv = torch.sum(torch.multiply(users_emb_inv, items_emb),dim=1)
        return self.act(scores_pri+scores_inv)