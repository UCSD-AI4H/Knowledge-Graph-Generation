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
    'used for': 'UsedFor'
}

def make_data_loader_atomic(triples):
    train_data = []
    hs,rs,ts = [],[],[]
    for triple in tqdm.tqdm(triples):
        h,r,t = triple
        h_id = tokenizer_gpt2.encode(h)
        r_id = tokenizer_gpt2.encoder[r]
        t_id = tokenizer_gpt2.encode(t)

        # if len(h_id) > 15 or len(t_id) > 15:
        #   # print('here')
        #   continue

        hs.append(h_id)
        rs.append(r_id)
        ts.append(t_id + [-1])

    hs_max = max([len(h) for h in hs])
    # rs_max = max([len(r) for r in rs])
    ts_max = max([len(t) for t in ts])
    for i in range(len(hs)):
        h = hs[i] + [50256] * (hs_max-len(hs[i]))
        # r = rs[i] + [50256] * (rs_max-len(rs[i]))
        r = [rs[i]]
        t = ts[i] + [50256] * (ts_max-len(ts[i]))

        # in atomic dataset, relation index is one integer
        sen = torch.LongTensor(h + r + t)
        attention_mask = (sen!=50256).long().tolist()
        
        idx = t.index(-1)
        t[idx] = 50256
        sen = h + r + t

        loss_mask = attention_mask
        loss_mask = torch.LongTensor(loss_mask)
        loss_mask[:hs_max+1] = 0
        loss_mask = loss_mask.tolist()
        generate_sen = h + r
        generate_mask = attention_mask[:hs_max+1]
        

        train_data.append((sen,generate_sen,attention_mask,loss_mask,generate_mask))

    return train_data

def make_data_loader_conceptnet(triples):
    train_data = []
  
    hs,rs,ts = [],[],[]
    for triple in tqdm.tqdm(triples):
        h,t,r = triple
        h_id = tokenizer_gpt2.encode(h)
        r_id = tokenizer_gpt2.encode(r)
        t_id = tokenizer_gpt2.encode(t)

        if len(h_id) > 10 or len(t_id) > 10:
            continue

        hs.append(h_id)
        rs.append(r_id)
        ts.append(t_id + [-1])

    hs_max = max([len(h) for h in hs])
    rs_max = max([len(r) for r in rs])
    ts_max = max([len(t) for t in ts])
    for i in range(len(hs)):
        h = hs[i] + [50256] * (hs_max-len(hs[i]))
        r = rs[i] + [50256] * (rs_max-len(rs[i]))
        t = ts[i] + [50256] * (ts_max-len(ts[i]))

        sen = torch.LongTensor(h + r + t)
        attention_mask = (sen!=50256).long().tolist()
        
        idx = t.index(-1)
        t[idx] = 50256
        sen = h + r + t

        loss_mask = attention_mask
        loss_mask = torch.LongTensor(loss_mask)
        loss_mask[:hs_max+rs_max] = 0
        loss_mask = loss_mask.tolist()
        generate_sen = h + r
        generate_mask = attention_mask[:hs_max+rs_max]
        

        train_data.append((sen,generate_sen,attention_mask,loss_mask,generate_mask))
        

    return train_data
  

if __name__ == "__main__":
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    if args.dataset == "conceptnet":
        # load entity2id and triples
        entity2id = pickle.load(open('conceptnet_entity2id','rb'))
        triple_list = pickle.load(open('conceptnet_triples','rb'))
        test_triples = pickle.load(open('conceptnet_test_triples','rb'))
        id2entity = dict([val,key] for key,val in entity2id.items())

        nodes_list_map = list(entity2id.keys())
        nodes_index_map = list(entity2id.values())
        nodes_num = len(entity2id)
        edges_num = len(triple_list)

        rel2text = dict([val,key] for key,val in text2rel.items())
        id2rel = {}
        id = 0
        for rel in list(rel2text.keys()):
            id2rel[id] = rel
            id += 1
        rel2id = dict([val,key] for key,val in id2rel.items())

        train_data = make_data_loader_conceptnet(triple_list)
        test_data = make_data_loader_conceptnet(test_triples)

        print("----------store dataset----------")
        pickle.dump(train_data,open('train_data_ckg_conceptnet','wb'))
        pickle.dump(test_data,open('test_data_ckg_conceptnet','wb'))
        print("train set:",len(train_data))
        print("test set:",len(test_data))
        print("----------done----------")
    else:
        # load entity2id and triples
        atomic_data = pickle.load(open('atomic_data','rb'))
        entity2id = pickle.load(open('atomic_entity2id','rb'))
        id2entity = dict([val,key] for key,val in entity2id.items())
        triple_list = [triple for triple in atomic_data['train']['total'] if triple[-1] != 'none']
        test_triples = [triple for triple in atomic_data['test']['total'] if triple[-1] != 'none']
        test_triples = test_triples[:72900]

        nodes_list_map = list(entity2id.keys())
        nodes_index_map = list(entity2id.values())
        nodes_num = len(entity2id)
        edges_num = len(triple_list)

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

        train_data = make_data_loader_atomic(triple_list)
        test_data = make_data_loader_atomic(test_triples)

        print("----------store dataset----------")
        pickle.dump(train_data,open('train_data_ckg_atomic','wb'))
        pickle.dump(test_data,open('test_data_ckg_atomic','wb'))
        print("train set:",len(train_data))
        print("test set:",len(test_data))
        print("----------done----------")














