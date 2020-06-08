#!/usr/bin/python3
# -*- coding: utf-8 -*-

# NECARE - NEtwork-based CAncer gene RElationship prediciton
#
# Written by Jiajun Qiu <jiajunqiu@hotmail.com>
#
# Copyright (c) 2018 Jiajun Qiu  <jiajunqiu@hotmail.com>


import torch
import pickle
import numpy as np
import math
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from necareMod.model import BaseLinear
from necareMod.model import BaseRGCN
import random
import necareMod.utils as utils
from optparse import OptionParser


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.h2=torch.arange(num_nodes).cuda()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(self.h2)
    
class RGCN(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=F.relu, self_loop=True,
                dropout=self.dropout)


    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)
    

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=True, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(2,h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r1 = self.w_relation[0]
        r2 = self.w_relation[1]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r1+r2 * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']    


def calc_final_score(embedding, w_relation,pairs):

    # DistMult
    s = embedding[pairs[:,0]]
    r1 = w_relation[0]
    r2 = w_relation[1]
    o = embedding[pairs[:,1]]
    score = torch.sum(s * r1+r2 * o, dim=1)
    score = torch.sigmoid(score)
    return score

dir_path = './'

#tmp_dir = tempfile.mkdtemp()

#Commandline parsing
disc = "NECARE - NEtwork-based CAncer gene RElationship prediciton"
usage = "usage: %prog [options]"+'\n\nExample:\nFor predict: NECARE.py -i ./dataset/test_pred.txt -o ./ \n'
usage = usage+'\nFor model training:  NECARE.py -t True -i ./dataset/test_trn.txt -g ./dataset/NECARE.graph -f ./dataset/NECARE_features.txt -s 0.1 -b 10 -e 10\n'
parser = OptionParser(usage=usage,description = disc)
parser.add_option("-t", action="store", type="string", dest="training", help="Turn on training model of NECARE to train your own modle(True/False), default is False")
parser.add_option("-i", action="store", type="string", dest="filename", help="Iutput file (tab-delimited text file) contains the pairs of input inetracitons. The first column for source genes, the second column for target genes. If training model is on, it need a third column for labels")
parser.add_option("-o", action="store", type="string", dest="path", help="path of the directory to save the prediction or trained model, defalt is current directory")
parser.add_option("-m", action="store", type="string", dest="model", help="The path of the modle for prediciton, default model is the one we reported in NECARE paper (if using default, parameters -m and -g will be ignored). -t True is incompatible with -m")
parser.add_option("-g", action="store", type="string", dest="graph", help="General gene relationship network,tab-delimited text file, the first column for source genes, the second column for target genes, the third column for inetraction types. Default is the one used in NECARE paper")
parser.add_option("-f", action="store", type="string", dest="feature", help="Features for the nodes, tab-delimited text file, the first column for gene names, Default is the what used in NECARE paper (OPA2Vec+TCGA)")
parser.add_option("-e", action="store", type="int", dest="epoch", help="Number of epoch (Only work for training model), default 100")
parser.add_option("-r", action="store", type="float", dest="learning_rate", help="Learning rate (Only work for training model), default 0.01")
parser.add_option("-l", action="store", type="int", dest="hidden_layer", help="Number of hidden layer (Only work for training model), default 2")
parser.add_option("-n", action="store", type="int", dest="hidden_node", help="Number of hidden node (Only work for training model), default 100")
parser.add_option("-d", action="store", type="float", dest="dropout", help="Rate of drapout (Only work for training model), default 0.2")
parser.add_option("-b", action="store", type="int", dest="base", help="Number of bases (Only work for training model), default 1")
parser.add_option("-s", action="store", type="float", dest="batch", help="Batch size (Only work for training model), default 0.2 (20% of general gene relationship network)")


options, args = parser.parse_args()


train = options.training
model = options.model 
input_file = options.filename
G = options.graph
output_path =  options.path
features_file = options.feature

num_epoch=options.epoch
learning_rate=options.learning_rate
num_bases=options.base
num_hidden_layers=options.hidden_layer
num_hidden_nodes=options.hidden_node
dropout=options.dropout
graph_batch_size=options.batch

if not input_file:
    parser.print_help()
    sys.exit()

if not output_path:
    output_path='./'

if not G:
    G = './dataset/NECARE.graph'

if not features_file:
    features_file = './dataset/NECARE_features.txt'

if train:
    if not num_epoch:
        num_epoch=100
    if not learning_rate:
        learning_rate=0.01
    if not num_hidden_layers:
        num_hidden_layers=2
    if not num_hidden_nodes:
        num_hidden_nodes=100
    if not dropout:
        dropout=0.2
    if not num_bases:
        num_bases=1
    if not graph_batch_size:
        graph_batch_size=0.2




if not train and not model:
    out=open('./necareMod/Models/node_dict.pkl','rb')
    node_dict=pickle.load(out)
    out.close()  
    out=open(output_path+'prediction.txt','w')
    try:
        omit=0
        input_pairs=[]
        input_pairs_names=[]
        for l in open(input_file):
            l=l.rstrip()
            t=l.split('\t')
            if t[0] in node_dict and t[1] in node_dict:
                input_pairs.append([node_dict[t[0]],node_dict[t[1]]])
                input_pairs_names.append([t[0],t[1]])
            else:
                omit+=1
        if omit>0:
            print('omit '+str(omit)+' pairs of input file') 
    except:
        print('Please check your input file. Must be a two column tab-delimited text file. The first column for source genes, the second column for target genes.')
        sys.exit()
    input_pairs=torch.tensor(input_pairs)
    scores=[]
    for fold in ('fold1','fold2','fold3','fold4','fold5'):
        node_embed=pickle.load(open('./necareMod/Models/'+fold+'_node.embed','rb')).cuda(0)
        edge_embed=pickle.load(open('./necareMod/Models/'+fold+'_edge.embed','rb')).cuda(0)
        score=calc_final_score(node_embed, edge_embed,input_pairs)
        scores.append(score)
    RI = (scores[0]+scores[1]+scores[2]+scores[3]+scores[4])/5
    RI = (2*RI-1)*100 
    pred = [1 if i >=0.5 else 0 for i in RI]
    for n in range(len(input_pairs_names)):
        print(input_pairs_names[n][0],input_pairs_names[n][1],int(RI[n]),pred[n],file=out,sep='\t')
        out.flush()
    out.close()
    sys.exit()

features={}
for l in open(features_file):
    l=l.rstrip()
    t=l.split('\t')
    features[t[0]]=t[1:]

edges_raw=[]
nodes_raw=[]
for l in open(G):
    l=l.rstrip()
    t=l.split('\t')
    nodes_raw.append(t[0])
    nodes_raw.append(t[1])
    edges_raw.append(t[2])

nodes=sorted(set(nodes_raw),key=nodes_raw.index)
edges=sorted(set(edges_raw),key=edges_raw.index)

num_edges=len(edges)
num_nodes=len(nodes)

nodes_idx=list(range(len(nodes)))
nodes_dict=dict(zip(nodes,nodes_idx))

edges_idx=list(range(len(edges)))
edges_dict=dict(zip(edges,edges_idx))


if not train:
    fea_data=[]
    for nod in nodes:
        fea_data.append(features[nod])
    fea_data=np.array(fea_data).astype('float16')
    fea_data=torch.from_numpy(fea_data).type('torch.FloatTensor')

    use_cuda=0
    model = LinkPredict(len(fea_data[0]), 100,  num_edges,  1,  2,  0.2, use_cuda, reg_param=0.01)
    try:
        checkpoint = torch.load(model)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    except:
        print('error with model loading')

    G_graph=[]
    for index, l in enumerate(open(G)):
        l=l.rstrip()
        t=l.split('\t')
        G_graph.append([nodes_dict[t[0]],edges_dict[t[2]],nodes_dict[t[1]]])
    G_graph=np.array(G_graph)
    test_graph, test_rel, test_norm = utils.build_test_graph(num_nodes, num_edges, np.array(G_graph))
#    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1,1)
#    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel).long()
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
    
    node_embed = model(test_graph, fea_data, test_rel, test_norm)
    edge_embed = model.w_relation

    try:
        omit=0
        input_pairs=[]
        input_pairs_names=[]
        for l in open(input_file):
            l=l.rstrip()
            t=l.split('\t')
            if t[0] in nodes_dict and t[1] in nodes_dict:
                input_pairs.append([nodes_dict[t[0]],nodes_dict[t[1]]])
                input_pairs_names.append([t[0],t[1]])
            else:
                omit+=1
        if omit>0:
            print('omit '+str(omit)+' pairs of input file') 
        input_pairs=torch.tensor(input_pairs)
    except:
        print('Please check your input file. Must be a two column tab-delimited text file. The first column for source genes, the second column for target genes.')
        sys.exit()


    RI=calc_final_score(node_embed, edge_embed,input_pairs)
    RI = (2*RI-1)*100 
    pred = [1 if i >=0.5 else 0 for i in RI]
    out=open(output_path+'prediction.txt','w')
    for n in range(len(input_pairs_names)):
        print(input_pairs_names[n][0],input_pairs_names[n][1],int(RI[n]),pred[n],file=out,sep='\t')
        out.flush()
    out.close()
    sys.exit()
else:
    try:
        omit=0
        input_pairs=[]
        input_pairs_names=[]
        input_nodes=[]
        for l in open(input_file):
            l=l.rstrip()
            t=l.split('\t')
            if t[0] in nodes_dict and t[1] in nodes_dict:
                input_pairs.append([nodes_dict[t[0]],int(t[2]),nodes_dict[t[1]]])
                input_pairs_names.append([t[0],t[1]])
                input_nodes.append(nodes_dict[t[0]])
                input_nodes.append(nodes_dict[t[1]])
            else:
                omit+=1
        if omit>0:
            print('omit '+str(omit)+' pairs of input file') 
        input_nodes=list(set(input_nodes))             
        input_pairs=torch.tensor(input_pairs)
    except:
        print('Please check your input file. Must be a three column tab-delimited text file. The first column for source genes, the second column for target genes and the third column for labels (1/0).')
        sys.exit()


    fea_data=[]
    for nod in nodes:
        fea_data.append(features[nod])
    fea_data=np.array(fea_data).astype('float16')
    fea_data=torch.from_numpy(fea_data).type('torch.FloatTensor')
    use_cuda=1
    model = LinkPredict(len(fea_data[0]), num_hidden_nodes,  num_edges,  num_bases,  num_hidden_layers,  dropout, use_cuda, reg_param=0.01)

    G_graph=[]
    input_edges=[]
    for index, l in enumerate(open(G)):
        l=l.rstrip()
        t=l.split('\t')
        G_graph.append([nodes_dict[t[0]],edges_dict[t[2]],nodes_dict[t[1]]])
        if [t[0],t[1]] in input_pairs_names:
            input_edges.append(index)
    G_graph=np.array(G_graph)
    graph_batch_size=int(graph_batch_size*len(G_graph))
    test_graph, test_rel, test_norm = utils.build_test_graph(num_nodes, num_edges, np.array(G_graph))
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel).long()
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
    
    model.cuda(device=torch.cuda.current_device())

    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, G_graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    forward_time = []
    backward_time = []

    # training loop
    print("start training...")
    graph_split_size=1
    edge_sampler='neighbor'
    epoch=0
    grad_norm=1.0

    while(epoch < num_epoch):
        epoch+=1
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data,data_fea,new_trn_nodes = \
                utils.generate_sampled_graph_and_labels(\
                G_graph,fea_data, graph_batch_size, graph_split_size,\
                num_edges, adj_list, degrees, input_edges,input_nodes,\
                edge_sampler)

        print("Done edge sampling")
        new_trn_samples=[]
        new_trn_labels=[]
        for xx in range(len(input_pairs)):
            p1=int(input_pairs[xx][0])
            p2=int(input_pairs[xx][2])
            if p1  in new_trn_nodes and  p2 in new_trn_nodes:
                new_trn_samples.append([p1,int(input_pairs[xx][1]),p2])
                new_trn_labels.append(int(input_pairs[xx][1]))   
        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type).long()
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)


        data_fea=data_fea.cuda(device=0)
        node_id, deg = node_id.cuda(device=0), deg.cuda(device=0)
        edge_type, edge_norm = edge_type.cuda(device=0), edge_norm.cuda(device=0)
            
        new_trn_samples=torch.from_numpy(np.array(new_trn_samples)).long().cuda(device=0)
        new_trn_labels=torch.from_numpy(np.array(new_trn_labels)).float().cuda(device=0)

        t0 = time.time()
        embed = model(g, data_fea, edge_type, edge_norm)
        loss = model.get_loss(g, embed, new_trn_samples, new_trn_labels)

        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f}  | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()  
    model_state_file = 'checkpoint.epoch'+str(epoch)
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,'loss': loss.item()},model_state_file)









'''


'''
