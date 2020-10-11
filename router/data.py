import torch
from transformers import BartTokenizer
import tqdm
from bart_evaluate import combine_into_words
import numpy as np
import pickle

language_dict = {combine_into_words[k]:k for k in combine_into_words.keys()}
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

mask = tokenizer.encode("<mask>")[1:-1][0]
end_id = tokenizer.encode("</s>")[1:-1][0]
start_id = tokenizer.encode("<s>")[1:-1][0]

def make_pretrain_dataset(setting,saved_data_pth = None,raw_data_pth = None,processed_data_pth = None):
    return 0


def make_finetune_dataset(saved_data_pth = None,raw_data_pth = None, processed_data_pth = None):
    raw_data = torch.load(raw_data_pth)
    train_data = raw_data["train"]
    test_data = raw_data["test_data"]
    dev_data = raw_data["dev"]
    if processed_data_pth != None:
        dataset = torch.load(processed_data_pth)
        return dataset,train_data,test_data

    def tokens2seqs(dataset):
      ret_dataset = []
      for s,r,o,_ in tqdm.tqdm(dataset):
        sr_ids = [start_id] + tokenizer.encode(s + ' ')[1:-1] + tokenizer.encode(r)[1:-1]
        o_ids = tokenizer.encode(o)[1:-1]
        input_ids = sr_ids + [end_id]
        output_ids = [start_id] + o_ids + [end_id]

        ret_dataset.append((input_ids,output_ids))

      return ret_dataset

    dataset = {}
    dataset["train"] = tokens2seqs(train_data)
    dataset["dev"] = tokens2seqs(dev_data)
    dataset["test"] = tokens2seqs(test_data)

    if saved_data_pth != None:
      torch.save(dataset,saved_data_pth)

    return dataset,train_data,test_data

    




def mege_to_sequence(lst_triples):
  seq = []
  for i in range(0,len(lst_triples)):
    seq.append(lst_triples[i][0])
    seq.append(language_dict[lst_triples[i][1]])
    if i == len(lst_triples) - 1:
      seq.append(lst_triples[i][2])
  return seq


def lst2ids(lst):
  ids = []
  for i,ents in enumerate(lst):
    if i != (len(lst) -1):
      ids += tokenizer.encode(ents + ' ')[1:-1]
      # ids += [1]
    else:
      ids += tokenizer.encode(ents)[1:-1]
  ids = [0] + ids + [2]

  return ids

def ids_deletion(ids,ratio):
  ids = ids[1:-1]

  deletion_number = int(len(ids) * ratio)

  for i in range(deletion_number):
      delete_id = np.random.randint(0,len(ids)-1)
      ids.pop(delete_id)

  ids = [0] + ids + [2]

  return ids

def ids_masking(ids,ratio):
  ids = ids[1:-1]

  masking_number = int(len(ids) * ratio)

  mask_ids = []
  for i in range(masking_number):
      mask_id = np.random.randint(0,len(ids)-1)
      while mask_id in mask_ids:
          mask_id = np.random.randint(0,len(ids)-1)
      ids[mask_id] = mask
      mask_ids.append(mask_id)

  ids = [0] + ids + [2]

  return ids

def ids_infilling(ids,num_of_span,lam):
    ids = ids[1:-1]

    span_len_lst = np.random.poisson(lam=lam, size=num_of_span)
    while sum(span_len_lst) >= len(ids):
        span_len_lst = np.random.poisson(lam=lam, size=num_of_span)

    orig_len_lst = np.random.rand(len(span_len_lst))
    ratio = (len(ids) - sum(span_len_lst)) / sum(orig_len_lst)
    orig_len_lst = orig_len_lst * ratio
    orig_len_lst = [int(x) for x in orig_len_lst]
    orig_len_lst.append((len(ids) - sum(span_len_lst)) - sum(orig_len_lst))

    ret_ids = []
    start_id = 0
    for i in range(len(span_len_lst)):
        span_len = span_len_lst[i]
        orig_len = orig_len_lst[i]
        ret_ids += ids[start_id:start_id + orig_len]
        ret_ids += [mask]
        start_id = start_id + span_len + orig_len

    ret_ids = [0] + ret_ids + [2]

    return ret_ids

    



