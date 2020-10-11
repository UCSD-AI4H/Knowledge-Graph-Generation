from transformers import BartForConditionalGeneration
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import time
import pickle

import copy

class Customized_Bart(nn.Module):
  def __init__(self,bart_model):
    super().__init__()
    self.shared = bart_model.model.shared
    self.encoder = bart_model.model.encoder
    self.decoder = bart_model.model.decoder
    self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

  def forward(self,args,input_ids, output_ids, encoder_attention_mask, decoder_attention_mask):
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


def make_model(model_pth = None):
  model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
  if model_pth != None:
        print("model loaded from {}".format(model_pth))
        model.load_state_dict(torch.load(model_pth))

  return model
















# def make_model(model_pth = None):
#     model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
#     if model_pth != None:
#         print("model loaded from {}".format(model_pth))
#         pretrain_model = Customized_Bart(model)
#         pretrain_model.load_state_dict(torch.load(model_pth)) 
#         model.encoder = copy.deepcopy(pretrain_model.encoder)
#         model.decoder = copy.deepcopy(pretrain_model.decoder)
#         # model.model.encoder = pretrain_model.encoder
#         # model.model.decoder = pretrain_model.decoder
#         model.shared = copy.deepcopy(pretrain_model.shared)
#         model.final_logits_bias = copy.deepcopy(pretrain_model.final_logits_bias)
    
#     return model