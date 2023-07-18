import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ccorr
import pandas as pd
from torch.nn import  Parameter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




class CompGraphConv(nn.Module):
    def __init__(
        self, in_dim, out_dim, comp_fn="sub", batchnorm=True, dropout=0.1
    ):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = th.tanh
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # define in/out/loop transform layer
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        # define relation transform layer
        self.W_R = nn.Linear(self.in_dim, self.out_dim)

        # self loop embedding
        self.loop_rel = nn.Parameter(th.Tensor(1, self.in_dim))
        nn.init.xavier_normal_(self.loop_rel)

    def forward(self, g, n_in_feats, r_feats):
        with g.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.
            g.srcdata["h"] = n_in_feats
            # print(n_in_feats.shape)
            # append loop_rel embedding to r_feats
            r_feats = th.cat((r_feats, self.loop_rel), 0)
            # print(r_feats.shape)
            # Assign features to all edges with the corresponding relation embeddings
            g.edata["h"] = r_feats[g.edata["etype"]] * g.edata["norm"]

            # Compute composition function in 4 steps
            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == "sub":
                g.apply_edges(fn.u_sub_e("h", "h", out="comp_h"))
            elif self.comp_fn == "mul":
                g.apply_edges(fn.u_mul_e("h", "h", out="comp_h"))
            elif self.comp_fn == "ccorr":
                g.apply_edges(
                    lambda edges: {
                        "comp_h": ccorr(edges.src["h"], edges.data["h"])
                    }
                )
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            # Step 2: use extracted edge direction to compute in and out edges
            comp_h = g.edata["comp_h"]

            in_edges_idx = th.nonzero(
                g.edata["in_edges_mask"], as_tuple=False
            ).squeeze()
            out_edges_idx = th.nonzero(
                g.edata["out_edges_mask"], as_tuple=False
            ).squeeze()
            # print(comp_h[out_edges_idx].shape)
            # print(comp_h.shape)
            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])

            new_comp_h = th.zeros(comp_h.shape[0], self.out_dim).to(
                comp_h.device
            )
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

            g.edata["new_comp_h"] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e("new_comp_h", "m"), fn.sum("m", "comp_edge"))

            # Step 4: add results of self-loop
            if self.comp_fn == "sub":
                comp_h_s = n_in_feats - r_feats[-1]
            elif self.comp_fn == "mul":
                comp_h_s = n_in_feats * r_feats[-1]
            elif self.comp_fn == "ccorr":
                comp_h_s = ccorr(n_in_feats, r_feats[-1])
            else:
                raise Exception("Only supports sub, mul, and ccorr")

            # Sum all of the comp results as output of nodes and dropout
            n_out_feats = (
                self.W_S(comp_h_s) + self.dropout(g.ndata["comp_edge"])
            ) * (1 / 3)

            # Compute relation output
            r_out_feats = self.W_R(r_feats)

            # Batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats, r_out_feats[:-1]


class CompGCN(nn.Module):
    def __init__(
        self,
        num_bases,
        num_rel,
        num_ent,
        in_dim=100,
        layer_size=[200],
        comp_fn="sub",
        batchnorm=True,
        dropout=0.1,
        layer_dropout=[0.3],
        userft_num = 14,
        eventft_num = 14,
        device = "cuda:1"
    ):
        super(CompGCN, self).__init__()

        self.num_bases = num_bases
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.in_dim = in_dim
        self.layer_size = layer_size
        self.comp_fn = comp_fn
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_layer = len(layer_size)

        self.userft_num = userft_num
        self.eventft_num = eventft_num
        self.user_map = nn.Linear(self.userft_num, self.in_dim)
        self.event_map = nn.Linear(self.eventft_num, self.in_dim)
        self.device = device
        self.dropout1 = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.activate = F.relu

        # CompGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(
            CompGraphConv(
                self.in_dim,
                self.layer_size[0],
                comp_fn=self.comp_fn,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
            )
        )
        for i in range(self.num_layer - 1):
            self.layers.append(
                CompGraphConv(
                    self.layer_size[i],
                    self.layer_size[i + 1],
                    comp_fn=self.comp_fn,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                )
            )

        # Initial relation embeddings
        if self.num_bases > 0:
            self.basis = nn.Parameter(th.Tensor(self.num_bases, self.in_dim))
            self.weights = nn.Parameter(th.Tensor(self.num_rel, self.num_bases))
            nn.init.xavier_normal_(self.basis)
            nn.init.xavier_normal_(self.weights)
        else:
            # event = pd.read_csv("./data/eventft_new.csv")
            # v = event.iloc[:,2:130]
            # # v = v.drop(["202112","202201","202202"],axis=1)
            # self.v = pd.concat([v,v],ignore_index=True).values
            # self.rel_embds = nn.Parameter(th.Tensor(self.v))
            # self.v = th.FloatTensor(self.v).to(self.device)
            self.rel_embds = nn.Parameter(th.Tensor(self.num_rel, self.in_dim))
            nn.init.xavier_normal_(self.rel_embds)
            # print(self.num_rel)

        # Node embeddings
        # user = pd.read_csv("./data/userft.csv")
        # self.u = user.iloc[:,1:].values
        # self.n_embds = th.Tensor(self.num_ent, v.shape[1]-self.u.shape[1])
        # self.u = th.FloatTensor(self.u).to(self.device)
        self.n_embds = nn.Parameter(th.Tensor(self.num_ent, self.in_dim))
        # self.n_embds = nn.Parameter(th.cat((th.Tensor(self.u),self.n_embds),dim=1))
        nn.init.xavier_normal_(self.n_embds)

        # Dropout after compGCN layers
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))

    def forward(self, graph):
        # node and relation features
        # 将数据通过线性层进行转换
        # new_v = self.event_map(self.v)
        # new_v = self.dropout1(new_v)
        # # self.rel_embds = nn.Parameter(th.Tensor(new_v))
        # # nn.init.xavier_normal_(self.rel_embds)
        # self.rel_embds = self.bn(new_v)
        # self.rel_embds = self.activate(self.rel_embds)

        # # 将数据通过线性层进行转换
        # new_u = self.user_map(self.u)
        # new_u = self.dropout1(new_u)
        # # self.n_embds = nn.Parameter(th.Tensor(new_u))
        # # nn.init.xavier_normal_(self.n_embds)
        # self.n_embds = self.bn(new_u)
        # self.n_embds = self.activate(self.n_embds)

        n_feats = self.n_embds
        if self.num_bases > 0:
            r_embds = th.mm(self.weights, self.basis)
            r_feats = r_embds
        else:
            r_feats = self.rel_embds

        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(graph, n_feats, r_feats)
            n_feats = dropout(n_feats)

        return n_feats, r_feats

class ConvE(nn.Module):
    def __init__(self, 
        num_rel,
        num_ent,
        in_dim=200,
        hide_size = 200,
        device = "cuda:0"):
        super(ConvE, self).__init__()
        self.device = device
        self.userft = pd.read_csv("data/userft_new.csv").iloc[:,1:].values
        self.eventft = pd.read_csv("data/eventft_new.csv").iloc[:,2:]
        self.eventft = pd.concat([self.eventft,self.eventft],ignore_index=True).values

        self.userft = th.FloatTensor(self.userft)
        self.eventft = th.FloatTensor(self.eventft)
        # self.userft = self.scaler_normalize(self.userft)
        # self.eventft = self.scaler_normalize(self.eventft)
        # self.userft = self.userft.to(self.device)
        # self.eventft = self.eventft.to(self.device)
        # self.ent_embs = nn.Embedding(num_ent, in_dim-self.userft.shape[1])
        # self.rel_embs = nn.Embedding(num_rel, in_dim-self.eventft.shape[1])
        self.linear1 = nn.Linear(self.userft.shape[1], in_dim)
        self.linear2 = nn.Linear(self.eventft.shape[1], in_dim)
        # self.user_embs = nn.Linear(self.userft.shape[1], in_dim)
        # self.edge_embs = nn.Linear(num_edge_ft, hidden_size)
        self.ent_embs = nn.Embedding(num_ent, in_dim)
        self.rel_embs = nn.Embedding(num_rel, in_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hide_drop = nn.Dropout(0.3)
        self.feature_drop = nn.Dropout2d(0.3)
        self.conv = nn.Conv2d(1, 32, (3, 3), bias=True)
        self.conv_mid = nn.Conv2d(1, 16, (3, 3), bias=True)
        self.bn_user = nn.BatchNorm1d(self.userft.shape[1])
        self.bn_rel = nn.BatchNorm1d(self.eventft.shape[1])
        self.bn0 = nn.BatchNorm2d(1)
        self.bn_mid = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(5760, in_dim)
        # self.fc = nn.Linear(5760, self.userft.shape[1])
        self.dim = in_dim #dim = 200
        self.dim1 = 16  #dim1 = 20
        self.dim2 = self.dim // self.dim1 # dim2 = 10
        self.register_parameter('b',Parameter(th.zeros(num_ent)))
        nn.init.xavier_normal_(self.ent_embs.weight.data)
        # nn.init.xavier_normal_(self.rel_embs.weight.data)
        nn.init.xavier_normal_(self.userft)
        nn.init.xavier_normal_(self.eventft)


        
    def forward(self, sub, rel):
        
        # e1_emb = self.ent_embs(sub).view(-1, 1, self.dim1, self.dim2)#el_emb; batch*1*20*10
        # rel_emb = self.rel_embs(rel).view(-1, 1 ,self.dim1, self.dim2)
        # e1_emb = self.ent_embs(sub)#el_emb; batch*1*20*10
        # rel_emb = self.rel_embs(rel)

        # e1_emb = th.cat([self.userft[sub],e1_emb],dim=1).view(-1, 1, self.dim1, self.dim2)
        # rel_emb = self.eventft[rel][:,:128].view(-1, 1, self.dim1, self.dim2)
        # userft = self.userft[sub]
        # e1_emb = self.linear1(self.input_drop(userft)).view(-1, 1, self.dim1, self.dim2)
        # e1_emb = self.bn0(e1_emb)
        # eventft = self.eventft[sub]
        # rel_emb = self.linear2(self.input_drop(eventft)).view(-1, 1, self.dim1, self.dim2)
        # rel_emb = self.bn0(rel_emb)
        e1_emb = sub.view(-1, 1, self.dim1, self.dim2)#el_emb; batch*1*20*10
        rel_emb = rel.view(-1, 1 ,self.dim1, self.dim2)
        conv_input = th.cat([e1_emb, rel_emb], dim = 2)#con_input: bath*1*40*10
        # print(conv_input.shape)
        conv_input = self.bn0(conv_input)
        x = self.input_drop(conv_input)
        # x = self.conv_mid(x)
        # x = self.bn_mid(x)
        # x = F.relu(x)
        # x = self.feature_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(x.shape[0], -1)#bacth*hide_size(38*8*32 = 9728)
        x = self.fc(x)
        x = self.hide_drop(x)
        x = self.bn2(x)
        x = F.relu(x)#batch*dim          ent_ems.weight   dim*ent_num
        #print(x.shape, self.ent_embs.weight.shape)
        # x = th.mm(x, (th.cat([self.userft,self.ent_embs.weight],dim=1)).transpose(1, 0))
        x = th.mm(x, self.ent_embs.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        score = th.sigmoid(x)
        return score

    def scaler_normalize(self,data):
        # 归一化
        data_min = data.min(dim=0)[0]
        data_max = data.max(dim=0)[0]
        data_norm = (data - data_min) / (data_max - data_min)
        return data_norm

class ConvE1(nn.Module):
    def __init__(self, 
        num_rel,
        num_ent,
        in_dim=200,
        hide_size = 200,
        device = "cuda:1"):
        super(ConvE, self).__init__()
        self.device = device
        # self.userft = pd.read_csv("data/userft.csv").iloc[:,1:].values
        # self.eventft = pd.read_csv("data/eventft.csv").iloc[:,2:]
        # self.eventft = pd.concat([self.eventft,self.eventft],ignore_index=True).values
        # 归一化
        # self.userft = th.from_numpy(self.userft)
        # self.eventft = th.from_numpy(self.eventft)
        # self.userft = self.userft.to(th.bfloat16)
        # self.eventft = self.eventft.to(th.bfloat16)
        # self.userft = th.FloatTensor(self.userft)
        # self.eventft = th.FloatTensor(self.eventft)
        # self.userft = self.scaler_normalize(self.userft)
        # self.eventft = self.scaler_normalize(self.eventft)
        # self.userft = self.userft.to(self.device)
        # self.eventft = self.eventft.to(self.device)
        self.ent_embs = nn.Embedding(num_ent, in_dim)
        self.rel_embs = nn.Embedding(num_rel, in_dim)
        # self.user_embs = nn.Linear(self.userft.shape[1], in_dim)
        # self.edge_embs = nn.Linear(num_edge_ft, hidden_size)
        self.input_drop = nn.Dropout(0.3)
        self.hide_drop = nn.Dropout(0.3)
        self.feature_drop = nn.Dropout2d(0.3)
        self.conv = nn.Conv2d(1, 32, (3, 3), bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(9728, in_dim)
        self.dim = in_dim #dim = 200
        self.dim1 = 16  #dim1 = 20
        self.dim2 = self.dim // self.dim1 # dim2 = 10
        self.register_parameter('b',Parameter(th.zeros(num_ent)))
        nn.init.xavier_normal_(self.ent_embs.weight.data)
        nn.init.xavier_normal_(self.rel_embs.weight.data)
        # self.init()
        
        
    # def init(self):
        
    def forward(self, sub, rel):
        
        e1_emb = self.ent_embs(sub).view(-1, 1, self.dim1, self.dim2)#el_emb; batch*1*20*10
        rel_emb = self.rel_embs(rel).view(-1, 1 ,self.dim1, self.dim2)
        # e1_emb = self.ent_embs(sub)#el_emb; batch*1*20*10
        # rel_emb = self.rel_embs(rel)
        # e1_emb = th.cat([self.userft[sub],e1_emb],dim=1).view(-1, 1, self.dim1, self.dim2)
        # rel_emb = th.cat([self.eventft[rel],rel_emb],dim=1).view(-1, 1, self.dim1, self.dim2)
        conv_input = th.cat([e1_emb, rel_emb], dim = 2)#con_input: bath*1*40*10
        conv_input = self.bn0(conv_input)
        x = self.input_drop(conv_input)
        x = self.conv(conv_input)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(x.shape[0], -1)#bacth*hide_size(38*8*32 = 9728)
        x = self.fc(x)
        x = self.hide_drop(x)
        x = self.bn2(x)
        x = F.relu(x)#batch*dim          ent_ems.weight   dim*ent_num
        #print(x.shape, self.ent_embs.weight.shape)
        x = th.mm(x, self.ent_embs.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        score = th.sigmoid(x)
        return score

    def scaler_normalize(self,data):
        # 归一化
        data_min = data.min(dim=0)[0]
        data_max = data.max(dim=0)[0]
        data_norm = (data - data_min) / (data_max - data_min)
        return data_norm


class CompGCN_DistMult(nn.Module):
    def __init__(self, 
        num_bases, 
        num_rel, 
        num_ent,
        in_dim,
        layer_size,
        comp_fn="sub",
        batchnorm=True,
        dropout=0.1,
        layer_dropout=[0.3],):
        super(CompGCN_DistMult, self).__init__()
        
        # compGCN model to get sub/rel embs
        self.compGCN_Model = CompGCN(
            num_bases,
            num_rel,
            num_ent,
            in_dim,
            layer_size,
            comp_fn,
            batchnorm,
            dropout,
            layer_dropout,
        )

        self.w_relation = nn.Parameter(th.Tensor(num_rel, layer_size[-1]))

        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, graph, sub, rel, dst=None):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        n_feats, r_feats = self.compGCN_Model(graph)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        if dst is None:
            x = th.mm(obj_emb, n_feats.transpose(1, 0))  # [batch_size, ent_num]
        else:
            dst_emb = n_feats[dst, :]
            x = th.sum(obj_emb*dst_emb, dim=1, keepdim=False)
            
        score = th.sigmoid(x)
        return score
