import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import OrderedDict, defaultdict
import pickle
import networkx as nx
import copy,math
import tqdm
import transformers
import dgl
from transformers import GPT2LMHeadModel,GPT2Tokenizer,GPT2Model
import dgl.function as fn
from pytorch_pretrained_bert import OpenAIAdam
import sys
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="conceptnet",
                    choices=["conceptnet","atomic"])
parser.add_argument("--subgraph_size", type=str, default="6")
parser.add_argument("--path_size", type=str, default="6")

args = parser.parse_args()

text2rel = {
    'at location': 'AtLocation',
    'capable of': 'CapableOf',
    'causes': 'Causes',
    'causes desire': 'CausesDesire',
    'created by': 'CreatedBy',
    'defined as': 'DefinedAs',
    'desire of': 'DesireOf',
    'desires': 'Desires',
    'has a': 'HasA',
    'has first subevent': 'HasFirstSubevent',
    'has last subevent': 'HasLastSubevent',
    'has pain character': 'HasPainCharacter',
    'has pain intensity': 'HasPainIntensity',
    'has prequisite': 'HasPrerequisite',
    'has property': 'HasProperty',
    'has subevent': 'HasSubevent',
    'inherits from': 'InheritsFrom',
    'instance of': 'InstanceOf',
    'is a': 'IsA',
    'located near': 'LocatedNear',
    'location of action': 'LocationOfAction',
    'made of': 'MadeOf',
    'motivated by goal': 'MotivatedByGoal',
    'not capable of': 'NotCapableOf',
    'not desires': 'NotDesires',
    'not has a': 'NotHasA',
    'not has property': 'NotHasProperty',
    'not is a': 'NotIsA',
    'not made of': 'NotMadeOf',
    'part of': 'PartOf',
    'receives action': 'ReceivesAction',
    'related to': 'RelatedTo',
    'symbol of': 'SymbolOf',
    'used for': 'UsedFor',
    'is self': 'Self'
}



def overall_graph(entity2id, triple_list):
    # constuct nx graph with conceptnet triples
    G = nx.DiGraph()
    nodes_list_map = list(entity2id.keys())
    nodes_index_map = [entity2id[i] for i in nodes_list_map]
    nodes_num = len(nodes_index_map)
    edges_num = len(triple_list)

    # add nodes
    for i in tqdm.tqdm(range(nodes_num)):
        entity = nodes_list_map[i]
        # embedding = torch.FloatTensor(data_triples.ent2emb[entity] if entity in data_triples.ent2emb else data_triples.ent_unk_emb).unsqueeze(0) 
        # embedding = gpt2_entity_embed[data_triples.word2idx[entity]].unsqueeze(0)                                 
        G.add_node(entity2id[entity],name=entity)

    # add edges 
    for i in tqdm.tqdm(range(edges_num)):
        if args.dataset == "conceptnet":
            h,t,r = triple_list[i]
        else:
            h,r,t = triple_list[i]
        # judge whether the other node is in the graph
        id1 = entity2id[h]
        id2 = entity2id[t]
        if id1 in nodes_index_map and id2 in nodes_index_map:
            # embedding = torch.FloatTensor(data_triples.rel2emb[triple[1]] if triple[1] in data_triples.rel2emb else data_triples.rel_unk_emb).unsqueeze(0)
            # embedding = gpt2_relation_embed[triple[1]].unsqueeze(0) 
            G.add_edge(id1,id2,name=r)
        else:
            print(id1,id2)
    return G

def construct_subgraph(G,source,length):
    '''given an entity node, find related subgraph'''
    G_adj = G.adj
    seen = set()
    nextlevel = {source}
    bfs_nodes = {}
    size = 0
    while nextlevel and size < length:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if size >= length:
                break
            if v not in seen:
                seen.add(v)
                if v == source:
                    adj = dict(G_adj[v]).keys()
                    adj = list(adj)
                    if len(adj) == 0:
                        continue
                    if len(adj) > length - 3:
                        # print(v,'here')
                        seed = min(4,np.random.randint(len(adj)))
                        seed = max(1,seed)
                    else:
                        seed = len(adj) - 1
                    # print(seed)
                    adj = np.random.choice(adj,size=seed)
                    new_adj = {}
                    for i in adj:
                        new_adj[i] = G_adj[v][i]
                # print(new_adj)
                else:
                    # randomly choice nodes to construct subgraph
                    adj = dict(G_adj[v]).keys()
                    adj = list(adj)
                    if len(adj) == 0:
                        continue
                    seed = min(3,np.random.randint(len(adj)))
                    seed = max(1,seed)
                    adj = np.random.choice(adj,size=seed)
                    new_adj = {}
                    for i in adj:
                        new_adj[i] = G_adj[v][i]
                size += len(new_adj)
                nextlevel.update(new_adj)
                bfs_nodes[v] = new_adj
    return bfs_nodes

def construct_path(G,source,length,subgraph):
    '''given a source node, find a path not overlapped with perviously found subgraph'''
    G_adj = G.adj
    seen = set(list(subgraph.keys()))
    dfs_path = [source]
    def DFS(G,v):
        seen.add(v)
        # yield v
        # print('v',v)
        # print(dfs_path)
        for w in G_adj[v]:
            # print(w,v)
            if len(dfs_path) < length:
                if w not in seen:
                    dfs_path.append(w)
                    DFS(G,w)
                    if len(dfs_path) >= length:
                        break
                    dfs_path.remove(dfs_path[-1])
                # else:
                #   dfs_path.remove(dfs_path[-1])
                #   print('1:',dfs_path)
                
                # adj = dict(G_adj[v]).keys()
                # adj = list(adj)
                # if w == adj[-1]:
                #   dfs_path.remove(dfs_path[-1])
                #   print('2:',dfs_path)
    DFS(G,source)

    return dfs_path

def get_norm_dict(ent_graph):
    ent_norm_dict = {}
    for nx_node_1 in ent_graph:
        graph_adj = ent_graph[nx_node_1]
        for nx_node_2 in graph_adj:
            if nx_node_2 not in ent_norm_dict:
                ent_norm_dict[nx_node_2] = {}
            rel_type = rel2id[G.edges[nx_node_1,nx_node_2]['name']]
            if rel_type not in ent_norm_dict[nx_node_2]:
                ent_norm_dict[nx_node_2][rel_type] = 1
            else:
                ent_norm_dict[nx_node_2][rel_type] += 1
    return ent_norm_dict


def construct_path_loop(G,source,length,subgraph):
    dfs_path = construct_path(G,source,length,subgraph)
    if length == len(dfs_path):
        return dfs_path
    else:
        dfs_path = construct_path(G,source,length-1,subgraph)
        if length - 1 == len(dfs_path):
            return dfs_path
        else:
            dfs_path = construct_path(G,source,length-2,subgraph)
            return dfs_path

def make_data_loader_atomic(G,entity_list,size_of_graph,length_of_path):
    '''generate the subgraph containing the given entity and path'''
    train_data = []
    graphes = []
    # node_features = [] # node_num * embed_size
    node_names = []
    nodes_num = [] # used for padding
    paths = []
    entities = []
    edge_t = []
    edge_n = []
    for entity in tqdm.tqdm(entity_list):
        # entity is id
        ent_graph = construct_subgraph(G,entity,size_of_graph)
        ent_path = construct_path_loop(G,entity,length_of_path,ent_graph)
        # clear the short path
        # print('---------------')
        # print('graph nodes:',data_triples.idx2word[entity],len(ent_graph))
        # print('path length:',len(ent_path))
        
        if len(ent_path) < length_of_path - 2:
            continue
        # transfrom to dgl graph
        g = dgl.DGLGraph()
        node_count = 0
        # ent_n_features = []
        ent_n_name = []
        ent2node = {}
        ent_count = []
        edge_count = 0
        edge_type = []
        edge_norm_dict = get_norm_dict(ent_graph)
        edge_norm_list = []
        for nx_node_1 in ent_graph:
            ent_count.append(nx_node_1)
            nx_node_adj = ent_graph[nx_node_1]
            # edge_count += len(nx_node_adj)
            # first check whether the node is already in dgl graph
            if nx_node_1 in ent2node:
                # print('here')
                src_node_count = ent2node[nx_node_1]
            else:
                # ent_embed = get_entity_embed(id2entity[nx_node_1]).unsqueeze(0)
                # ent_n_features.append(ent_embed.detach())
                # ent_n_name.append(torch.LongTensor(entity2id[G.nodes[nx_node_1]['name']]).unsqueeze(0))
                ent_n_name.append(entity2id[G.nodes[nx_node_1]['name']])
                g.add_nodes(1)
                ent2node[nx_node_1] = node_count # store entity's corresponding node is in dgl graph
                src_node_count = node_count
                node_count += 1
            for nx_node_2 in nx_node_adj:
                ent_count.append(nx_node_2)
                # first check whether the node is already in dgl graph
                if nx_node_2 in ent2node:
                    # already in graph
                    # print('here')
                    g.add_edges(src_node_count,ent2node[nx_node_2])
                    rel_type = rel2id[G.edges[nx_node_1,nx_node_2]['name']]
                    edge_type.append(torch.LongTensor([rel_type]).unsqueeze(0))
                    edge_norm_list.append(torch.FloatTensor([1 / edge_norm_dict[nx_node_2][rel_type]]).unsqueeze(0))
                    edge_count += 1
                else:
                    # store features
                    # ent_embed = get_entity_embed(id2entity[nx_node_2]).unsqueeze(0)
                    # ent_n_features.append(ent_embed.detach())
                    # ent_n_name.append(torch.LongTensor(entity2id[G.nodes[nx_node_1]['name']]).unsqueeze(0))
                    ent_n_name.append(entity2id[G.nodes[nx_node_2]['name']])
                    # add node and edge to dgl graph
                    g.add_nodes(1)
                    ent2node[nx_node_2] = node_count
                    g.add_edges(src_node_count,node_count)
                    node_count += 1
                    edge_count += 1
                    rel_type = rel2id[G.edges[nx_node_1,nx_node_2]['name']]
                    # print(rel_type)
                    edge_type.append(torch.LongTensor([rel_type]).unsqueeze(0))
                    edge_norm_list.append(torch.FloatTensor([1 / edge_norm_dict[nx_node_2][rel_type]]).unsqueeze(0))

        # # add self loop for rgcn
        # for idx in range(g.number_of_nodes()):
        #   g.add_edges(idx,idx)
        #   edge_count += 1
        #   edge_type.append(torch.LongTensor([len(rel2id)]).unsqueeze(0))
        #   edge_norm_list.append(torch.FloatTensor([1]).unsqueeze(0))
        
        # assertion
        assert(g.number_of_nodes()==len(set(ent_count))) # check no extra node added
        assert(g.number_of_edges()==edge_count) # check no extra edge added
        assert(g.number_of_nodes()==node_count)
        
        if node_count < size_of_graph - 2:
            continue
        if node_count > 25:
            print(entity, node_count)
        

        # try to add features directly to nodes, without padding
        # g.ndata['h'] = torch.cat(ent_n_features,dim=0) # [node_num, hidden]
        # g.edata['rel_type'] = torch.cat(edge_type,dim=0) # [edge_num, 1]
        # g.edata['norm'] = torch.cat(edge_norm_list,dim=0) # [edge_num, 1] 

        # ent_n_features = torch.cat(ent_n_features,dim=0)
        edge_type = torch.cat(edge_type,dim=0)
        edge_norm_list = torch.cat(edge_norm_list,dim=0)
        # ent_n_names = torch.cat(ent_n_name,dim=0)

        # may cause OOM, try padding  
        # node_features.append(ent_n_features)
        node_names.append(ent_n_name)
        edge_t.append(edge_type)
        edge_n.append(edge_norm_list)
        entities.append(entity)
        graphes.append(g)
        nodes_num.append(node_count)
        # for path, we need e + r + e + r... in sentence format
        path_text = []
        # print(ent_path)
        for i in range(len(ent_path)-1):
            nx_node_1 = ent_path[i]
            nx_node_2 = ent_path[i+1]
            path_text.extend(tokenizer_gpt2.encode(G.nodes[nx_node_1]['name']))

            # rel = G.edges[nx_node_1,nx_node_2]
            # print(nx_node_1,nx_node_2)
            # print(G.edges[nx_node_1,nx_node_2]['name'])
            # for relation, use encoder dictonary
            # print(G.edges[nx_node_1,nx_node_2]['name'])
            # a = G.edges[nx_node_1,nx_node_2]['name']
            # b = tokenizer_gpt2.encoder[a]
            # print(b) 
            # print(path_text)
            path_text.append(tokenizer_gpt2.encoder[G.edges[nx_node_1,nx_node_2]['name']])
        # add the last entity
        path_text.extend(tokenizer_gpt2.encode(G.nodes[ent_path[-1]]['name']))
        paths.append(path_text)

    # padding
    nodes_pad_num = max(nodes_num)
    path_pad_num = 40
    print('data pieces:',len(graphes))
    for i in tqdm.tqdm(range(len(graphes))):
        g = graphes[i]
        # g.add_nodes(nodes_pad_num-nodes_num[i])
        # zeros = torch.zeros(768).unsqueeze(0)

        # node_feature_g = node_features[i]
        node_name_g = node_names[i]
        edge_rel_type = edge_t[i]
        edge_rel_norm = edge_n[i]
        # node_feature_g += [zeros] * (nodes_pad_num-nodes_num[i])
        # node_feature_g = torch.cat(node_feature_g)

        new_path = [50256] + paths[i] + [50256]

        # print('new_path_ids:',new_path)
        new_path_len = len(new_path)
        new_path += [50256] * (path_pad_num - new_path_len)
        
        # graph_mask = [1] * nodes_num[i] + [0] * (nodes_pad_num-nodes_num[i])
        path_mask = [1] * new_path_len + [0] * (path_pad_num - new_path_len)

        # add to list
        train_data.append((g,node_name_g,edge_rel_type,edge_rel_norm,new_path,path_mask,entities[i]))
        # train_data.append((g,new_path,path_mask,entities[i]))
        # assert(g.number_of_nodes() == node_feature_g.shape[0])
        assert(len(path_mask) == len(new_path))

    return train_data, nodes_num

def make_data_loader_conceptnet(G,entity_list,size_of_graph,length_of_path):
    '''generate the subgraph containing the given entity and path'''
    train_data = []
    graphes = []
    # node_features = [] # node_num * embed_size
    node_names = []
    nodes_num = [] # used for padding
    paths = []
    entities = []
    edge_t = []
    edge_n = []
    for entity in tqdm.tqdm(entity_list):
        # entity is id
        ent_graph = construct_subgraph(G,entity,size_of_graph)
        ent_path = construct_path_loop(G,entity,length_of_path,ent_graph)
        # clear the short path
        # print('---------------')
        # print('graph nodes:',data_triples.idx2word[entity],len(ent_graph))
        # print('path length:',len(ent_path))
    
        if len(ent_path) < length_of_path - 2:
            continue
        # transfrom to dgl graph
        g = dgl.DGLGraph()
        node_count = 0
        # ent_n_features = []
        ent_n_name = []
        ent2node = {}
        ent_count = []
        edge_count = 0
        edge_type = []
        edge_norm_dict = get_norm_dict(ent_graph)
        edge_norm_list = []
        for nx_node_1 in ent_graph:
            ent_count.append(nx_node_1)
            nx_node_adj = ent_graph[nx_node_1]
            # edge_count += len(nx_node_adj)
            # first check whether the node is already in dgl graph
            if nx_node_1 in ent2node:
                # print('here')
                src_node_count = ent2node[nx_node_1]
            else:
                # ent_embed = get_entity_embed(id2entity[nx_node_1]).unsqueeze(0)
                # ent_n_features.append(ent_embed.detach())
                # ent_n_name.append(torch.LongTensor(entity2id[G.nodes[nx_node_1]['name']]).unsqueeze(0))
                ent_n_name.append(entity2id[G.nodes[nx_node_1]['name']])
                g.add_nodes(1)
                ent2node[nx_node_1] = node_count # store entity's corresponding node is in dgl graph
                src_node_count = node_count
                node_count += 1
            for nx_node_2 in nx_node_adj:
                ent_count.append(nx_node_2)
                # first check whether the node is already in dgl graph
                if nx_node_2 in ent2node:
                    # already in graph
                    # print('here')
                    g.add_edges(src_node_count,ent2node[nx_node_2])
                    rel_type = rel2id[G.edges[nx_node_1,nx_node_2]['name']]
                    edge_type.append(torch.LongTensor([rel_type]).unsqueeze(0))
                    edge_norm_list.append(torch.FloatTensor([1 / edge_norm_dict[nx_node_2][rel_type]]).unsqueeze(0))
                    edge_count += 1
                else:
                    # store features
                    # ent_embed = get_entity_embed(id2entity[nx_node_2]).unsqueeze(0)
                    # ent_n_features.append(ent_embed.detach())
                    # ent_n_name.append(torch.LongTensor(entity2id[G.nodes[nx_node_1]['name']]).unsqueeze(0))
                    ent_n_name.append(entity2id[G.nodes[nx_node_2]['name']])
                    # add node and edge to dgl graph
                    g.add_nodes(1)
                    ent2node[nx_node_2] = node_count
                    g.add_edges(src_node_count,node_count)
                    node_count += 1
                    edge_count += 1
                    rel_type = rel2id[G.edges[nx_node_1,nx_node_2]['name']]
                    # print(rel_type)
                    edge_type.append(torch.LongTensor([rel_type]).unsqueeze(0))
                    edge_norm_list.append(torch.FloatTensor([1 / edge_norm_dict[nx_node_2][rel_type]]).unsqueeze(0))

        # add self loop for rgcn
        for idx in range(g.number_of_nodes()):
            g.add_edges(idx,idx)
            edge_count += 1
            edge_type.append(torch.LongTensor([len(rel2id)]).unsqueeze(0))
            edge_norm_list.append(torch.FloatTensor([1]).unsqueeze(0))
        
        # assertion
        assert(g.number_of_nodes()==len(set(ent_count))) # check no extra node added
        assert(g.number_of_edges()==edge_count) # check no extra edge added
        assert(g.number_of_nodes()==node_count)
        
        if node_count < size_of_graph - 2:
            continue
        if node_count > 25:
            print(entity, node_count)

        # try to add features directly to nodes, without padding
        # g.ndata['h'] = torch.cat(ent_n_features,dim=0) # [node_num, hidden]
        # g.edata['rel_type'] = torch.cat(edge_type,dim=0) # [edge_num, 1]
        # g.edata['norm'] = torch.cat(edge_norm_list,dim=0) # [edge_num, 1] 

        # ent_n_features = torch.cat(ent_n_features,dim=0)
        edge_type = torch.cat(edge_type,dim=0)
        edge_norm_list = torch.cat(edge_norm_list,dim=0)
        # ent_n_names = torch.cat(ent_n_name,dim=0)

        # may cause OOM, try padding  
        # node_features.append(ent_n_features)
        node_names.append(ent_n_name)
        edge_t.append(edge_type)
        edge_n.append(edge_norm_list)
        entities.append(entity)
        graphes.append(g)
        nodes_num.append(node_count)
        # for path, we need e + r + e + r... in sentence format
        path_text = []
        # print(ent_path)
        for i in range(len(ent_path)-1):
            nx_node_1 = ent_path[i]
            nx_node_2 = ent_path[i+1]
            path_text.append(G.nodes[nx_node_1]['name'])

            # rel = G.edges[nx_node_1,nx_node_2]
            # print(nx_node_1,nx_node_2)
            # print(G.edges[nx_node_1,nx_node_2]['name'])

            path_text.extend(rel2text[G.edges[nx_node_1,nx_node_2]['name']].split())
        # add the last entity
        path_text.append(G.nodes[ent_path[-1]]['name'])
        paths.append(path_text)

    # padding
    nodes_pad_num = max(nodes_num)
    path_pad_num = 50
    # print('data pieces:',len(graphes))


    for i in tqdm.tqdm(range(len(graphes))):
        g = graphes[i]
        # g.add_nodes(nodes_pad_num-nodes_num[i])
        # zeros = torch.zeros(768).unsqueeze(0)

        # node_feature_g = node_features[i]
        node_name_g = node_names[i]
        edge_rel_type = edge_t[i]
        edge_rel_norm = edge_n[i]
        # node_feature_g += [zeros] * (nodes_pad_num-nodes_num[i])
        # node_feature_g = torch.cat(node_feature_g)

        new_path = ['<|endoftext|>'] + paths[i] + ['<|endoftext|>']
        # new_path = paths[i] + ['<|endoftext|>']

        # new_path = ['[CLS]'] + paths[i] + ['[SEP]']
        
        # print('new_path_ids:',' '.join(new_path))
        new_path = ' '.join(new_path)
        # print('new_path:',new_path)
        new_path = tokenizer_gpt2.encode(new_path)
        # print('new_path_ids:',new_path)
        new_path_len = len(new_path)
        new_path += [50256] * (path_pad_num - new_path_len)
        
        # graph_mask = [1] * nodes_num[i] + [0] * (nodes_pad_num-nodes_num[i])
        path_mask = [1] * new_path_len + [0] * (path_pad_num - new_path_len)

        # add to list
        train_data.append((g,node_name_g,edge_rel_type,edge_rel_norm,new_path,path_mask,entities[i]))
        # train_data.append((g,new_path,path_mask,entities[i]))
        # assert(g.number_of_nodes() == node_feature_g.shape[0])
        assert(len(path_mask) == len(new_path))

    return train_data, nodes_num


if __name__ == "__main__":
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    # load entity2id and triples
    if args.dataset == "conceptnet":
        entity2id = pickle.load(open('data/conceptnet_entity2id.pkl','rb'))
        triple_list = pickle.load(open('data/conceptnet_triples.pkl','rb'))
        id2entity = dict([val,key] for key,val in entity2id.items())

        if os.path.exists("data/conceptnet_graph.pkl"):
            print("----------load existing graph----------")
            G = pickle.load(open("data/conceptnet_graph.pkl","rb"))
        else:
            print("----------construct graph----------")
            G = overall_graph(entity2id, triple_list)
            pickle.dump(G,open('data/conceptnet_graph.pkl','wb'))

        nodes_list_map = list(entity2id.keys())
        nodes_list_map = nodes_list_map
        nodes_index_map = [entity2id[i] for i in nodes_list_map]
        nodes_num = len(nodes_index_map)
        edges_num = len(triple_list)
        rel2text = dict([val,key] for key,val in text2rel.items())
        id2rel = {}
        id = 0
        for rel in list(rel2text.keys()):
            id2rel[id] = rel
            id += 1
        rel2id = dict([val,key] for key,val in id2rel.items())
        subgraph_size = int(args.subgraph_size)
        path_size = int(args.path_size)

        print("----------construct conceptnet dataset----------")
        train_data, num_train_data = make_data_loader_conceptnet(G,nodes_index_map[:-1500],subgraph_size,path_size)
        test_data, num_test_data = make_data_loader_conceptnet(G,nodes_index_map[-1500:],subgraph_size,path_size)
        print("train set:",len(num_train_data))
        print("test set:",len(num_test_data))
        print("----------save dataset----------")
        pickle.dump(train_data,open('data/train_data_rgcn_conceptnet.pkl','wb'))
        pickle.dump(test_data,open('data/test_data_rgcn_conceptnet.pkl','wb'))
        print("----------done----------")
    else:
        # load ATOMIC tuples
        atomic_data = pickle.load(open('data/atomic_data.pkl','rb'))
        entity2id = pickle.load(open('data/atomic_entity2id.pkl','rb'))
        id2entity = dict([val,key] for key,val in entity2id.items())
        triple_list = [triple for triple in atomic_data['train']['total'] if triple[-1] != 'none']
        test_triples = [triple for triple in atomic_data['test']['total'] if triple[-1] != 'none']

        if os.path.exists("data/atomic_graph.pkl"):
            print("----------load existing graph----------")
            G = pickle.load(open("data/atomic_graph.pkl","rb"))
        else:
            print("----------construct graph----------")
            G = overall_graph(entity2id, triple_list)
            pickle.dump(G,open('data/atomic_graph.pkl','wb'))

        rels = ["<oReact>", "<oEffect>", "<oWant>", "<xAttr>", "<xEffect>", "<xIntent>", "<xNeed>", "<xReact>", "<xWant>"]
        count = 50257
        for rel in rels:
            tokenizer_gpt2.encoder[rel] = count
            tokenizer_gpt2.decoder[count] = rel
            count += 1
        id2rel = {}
        id = 0
        for rel in list(rels):
            id2rel[id] = rel
            id += 1
        rel2id = dict([val,key] for key,val in id2rel.items())

        nodes_list_map = list(entity2id.keys())
        nodes_index_map = [entity2id[i] for i in nodes_list_map]
        nodes_num = len(nodes_index_map)
        edges_num = len(triple_list)

        subgraph_size = int(args.subgraph_size)
        path_size = int(args.path_size)

        print("----------construct atomic dataset----------")
        train_data, nodes_pad_num = make_data_loader_atomic(G,nodes_index_map,subgraph_size,path_size)
        test_data = train_data[-1000:]
        train_data = train_data[:-1000]
        print("train set:",len(train_data))
        print("test set:",len(test_data))
        print("----------save dataset----------")
        pickle.dump(train_data,open('data/train_data_rgcn_atomic.pkl','wb'))
        pickle.dump(test_data,open('data/test_data_rgcn_atomic.pkl','wb'))
        print("----------done----------")


    



