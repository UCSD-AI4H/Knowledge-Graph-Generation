from pytorch_pretrained_bert import OpenAIAdam
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
import argparse



class Customized_Bart(nn.Module):
  def __init__(self,bart_model):
    super().__init__()
    self.shared = bart_model.model.shared
    self.encoder = bart_model.model.encoder
    self.decoder = bart_model.model.decoder
    self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

  def forward(self,input_ids, output_ids, encoder_attention_mask, decoder_attention_mask):
    batch_size = args.batch_size
    token_sum_length = encoder_attention_mask.reshape(batch_size,-1,30).sum(1) #(bsz,30)
    token_sum_length = token_sum_length.masked_fill(token_sum_length == 0, 1) # To avoid devide zero
    encoder_hidden_states = self.encoder(input_ids, attention_mask= encoder_attention_mask) #(bsz*5,30,1024)
    encoder_hidden_states = encoder_hidden_states[0]

    encoder_hidden_states = encoder_hidden_states.masked_fill((encoder_attention_mask == 0).unsqueeze(-1), 0)
    encoder_hidden_states = encoder_hidden_states.reshape(batch_size,-1,30,1024)
    avg_encoder_hidden_states = encoder_hidden_states.sum(1) / token_sum_length.unsqueeze(-1) #(bsz,30,1024)
    output_ids, decoder_padding_mask, decoder_causal_mask = prepare_bart_decoder_inputs(decoder_input_ids = output_ids)
    avg_encoder_attention_mask = encoder_attention_mask.reshape(batch_size,-1,30).sum(1)
    avg_encoder_attention_mask = avg_encoder_attention_mask.masked_fill(avg_encoder_attention_mask != 0, 1)
    x=  self.decoder(output_ids,avg_encoder_hidden_states,avg_encoder_attention_mask,decoder_padding_mask,decoder_causal_mask)
    lm_logits = F.linear(x[0], self.shared.weight, bias=self.final_logits_bias)
    return lm_logits


def prepare_bart_decoder_inputs(decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def main(args):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    epochs=args.epoch_num
    num_gradients_accumulation=args.num_gradients_accumulation
    batch_size=args.batch_size
    gpu_id=0
    lr=args.lr
    # load_dir='decoder_model'
    device = torch.device("cuda:0")
    MAX_ENCODER_LEN = 30
    MAX_DECODER_LEN = 200

    train_data = torch.load(args.train_data) 

    train_data = train_data[0:54336]

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=16)
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model = Customized_Bart(bart_model).to(device)

    # optimizer
    num_train_optimization_steps = len(train_data) * epochs // batch_size // num_gradients_accumulation

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
          
          input_ids = input_ids.reshape(batch_size*10,30)
          input_mask= input_mask.reshape(batch_size*10,30)
          output_ids = output_ids.reshape(batch_size*1,30)
          output_mask= output_mask.reshape(batch_size*1,30)
          logits = model(input_ids, output_ids,input_mask,output_mask)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="conceptnet")
    parser.add_argument("--method", type=str, default="pretrain")
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_gradients_accumulation", type=int, default= 2)

    parser.add_argument("--batch_size", type=int, default= 16)
    parser.add_argument("--train_data", type=str, default="train_data_10_inputs.pth")
    parser.add_argument("--test_data", type=str, default="data/test_data_rgcn_conceptnet")
    parser.add_argument("--model_file", type=str, default="models/Bart_10.pkl")
    parser.add_argument("--log_file", type=str, default="log/new_log")
    parser.add_argument("--load_model", action="store_true")

    args = parser.parse_args()
    main(args)



