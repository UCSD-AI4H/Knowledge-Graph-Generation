from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

import torch
import pickle
df=open('train100k.txt','r')
lines=df.readlines()
df.close()

dict_ = {
    'FormOf':"is form of",
    'AtLocation': "is at location",
    'CapableOf': "is capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "is created by",
    'DefinedAs': "is defined as",
    'DesireOf': "is desire of",
    'Desires': "desires",
    'HasA': "has a",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'is instance of',
    'IsA': "is a",
    'LocatedNear': "is located near",
    'LocationOfAction': "is location of action",
    'MadeOf': "is made of",
    'MotivatedByGoal': "is motivated by goal",
    'NotCapableOf': "is not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "is not made of",
    'PartOf': "is part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "is related to",
    'SymbolOf': "is symbol of",
    'UsedFor': "is used for"}

datas = []
for line in lines:
    data = line.split("\t")
    datas.append(data[:-1])

head_ = []
rel_ = []
tail_ = []
for i in range (len(datas)):
    head_.append(datas[i][1])
    rel_.append(dict_[datas[i][0]])
    tail_.append(datas[i][2])

c = list(set(head_).intersection(set(tail_)))

def find_index(List,elem):
#"List represents List， i represents index value，v represents values，elem represents the value we need to fine """
    return [i for (i,v) in enumerate(List) if v==elem]


processed_data = []
a = []
for concept in c:
    input_triples = []
    tail_i = find_index(tail_,concept)
    for i in tail_i:
        input_triples.append(tokenizer.encode(head_[i]+" "+rel_[i]+" "+tail_[i])) 
    head_i = find_index(head_,concept)
    for i in head_i:
        output_triple =tokenizer.encode( head_[i]+" "+rel_[i]+" "+tail_[i])
        processed_data.append((input_triples,output_triple))

torch.save(processed_data,"datas/processed.pkl")