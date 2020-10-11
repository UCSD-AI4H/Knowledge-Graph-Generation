from pytorch_pretrained_bert import OpenAIAdam
from model import Customized_Bart
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import time
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
import os




def get_optimizer(model, epochs, train_data_length, batch_size, num_gradients_accumulation=2, lr=1e-5):
     # optimizer
    num_train_optimization_steps = train_data_length * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=lr,
                        warmup=0.01,
                        max_grad_norm=1.0,
                        weight_decay=0.01,
                        t_total=num_train_optimization_steps)
                        # schedule=opt.train.static.lrsched)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 1)
    return optimizer,criterion

def train(args, model, device, train_dataloader, batch_size, epochs, optimizer, criterion, num_gradients_accumulation,max_encoder_length):
    update_count = 0
    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        for batch in tqdm.tqdm(train_dataloader):
          batch = [item.to(device) for item in batch]

          input_ids,input_mask,output_ids,output_mask= batch
          
          input_ids = input_ids.reshape(batch_size*10,max_encoder_length)
          input_mask= input_mask.reshape(batch_size*10,max_encoder_length)
          output_ids = output_ids.reshape(batch_size*1,max_encoder_length)
          output_mask= output_mask.reshape(batch_size*1,max_encoder_length)
          logits = model(args,input_ids, output_ids,input_mask,output_mask)
          logits = logits

          out = logits[:, :-1].contiguous()
          target = output_ids[:, 1:].contiguous()
          out = out.reshape(-1, out.shape[-1])
          target = target.reshape(-1)
          loss = criterion(out, target)
          loss.backward()

          losses += loss.item()
          times += 1
          update_count += 1
          if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
              optimizer.step()
              optimizer.zero_grad()

        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        start = end
        torch.save(model.state_dict(), args.model_file)
        
