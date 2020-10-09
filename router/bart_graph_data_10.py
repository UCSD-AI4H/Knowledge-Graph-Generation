import torch
input = torch.load("processed_dataset_.pkl")

count = 0
input_ =[]
for p in input:
  if len(p[0])>=11:
    count+=1
    input_.append(p)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import time
import pickle
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

import torch
from torch.utils.data import TensorDataset, DataLoader
epochs=10
num_gradients_accumulation=1
batch_size=2
gpu_id=0
lr=1e-5
# load_dir='decoder_model'
device = torch.device("cuda:0")
MAX_ENCODER_LEN = 30
MAX_DECODER_LEN = 200

train_data = []
for data in input_:
  input_id = data[0]
  output_id = data[-1]
  output_mask = [1]*(len(output_id))
  output_mask += [0] * (MAX_ENCODER_LEN - len(output_id))
  output_id += [1] * (MAX_ENCODER_LEN - len(output_id))
  output_id = torch.LongTensor(output_id)
  output_mask = torch.LongTensor(output_mask)

  input_mask = []
  for i in range (len(input_id)):
    mask = [1]*(len(input_id[i]))
    mask += [0] * (MAX_ENCODER_LEN - len(input_id[i]))
    input_mask.append(mask)
    input_id[i] += [1] * (MAX_ENCODER_LEN - len(input_id[i]))
  
  i = len(input_id)
  k = 0
  while i//10>0:
    input_split = []
    mask_split = []
    input_split.append(input_id[k*10:k*10+10])
    mask_split.append(input_mask[k*10:k*10+10])
    k += 1
    i -= 10
    # turn to tensor 
    encoder_in = torch.LongTensor(input_split)
    encoder_in = encoder_in.squeeze()
    mask_encoder_in = torch.LongTensor(mask_split)
    mask_encoder_in = mask_encoder_in.squeeze()
    train_data.append((encoder_in,mask_encoder_in,output_id,output_mask))

len(train_data)

train_data[1]

torch.save(train_data, 'train_data_10_inputs.pth')