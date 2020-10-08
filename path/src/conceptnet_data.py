import torch
from transformers import BartTokenizer
import tqdm
from bart_evaluate import combine_into_words
import numpy as np
import pickle
import os

language_dict = {combine_into_words[k]:k for k in combine_into_words.keys()}
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

mask = tokenizer.encode("<mask>")[1:-1][0]
end_id = tokenizer.encode("</s>")[1:-1][0]
start_id = tokenizer.encode("<s>")[1:-1][0]

def make_conceptnet_pretrain_dataset(args,setting):
    
    if args.data_name == "path2path":
        processed_data_pth = "data/processed_pretrain_path2path_data.pkl"
        raw_data_pth = "data/pretrain_path2path_data.pkl"
        saved_data_pth = "data/processed_pretrain_path2path_data.pkl"

        if os.path.exists(processed_data_pth):
            dataset = torch.load(processed_data_pth)
        else:
            dataset = process_pretrain_path2path_raw_data(raw_data_pth,saved_data_pth)
    
    if args.data_name == "path":
        processed_data_pth = "data/processed_pretrain_path_data.pkl"
        raw_data_pth = "data/pretrain_path_data.pkl"
        saved_data_pth = "data/processed_pretrain_path_data.pkl"

        if os.path.exists(processed_data_pth):
            dataset = torch.load(processed_data_pth)
        else:
            dataset = process_pretrain_path_raw_data(setting, raw_data_pth,saved_data_pth)


    ret_dataset = {}
    if args.toy:
        ret_dataset["train"] = dataset[:500]
    else:
        ret_dataset["train"] = dataset[:100000]
    ret_dataset["test"] = dataset[100000:102000]
    return ret_dataset


def make_conceptnet_finetune_dataset(saved_data_pth = None,raw_data_pth = None, processed_data_pth = None, toy = False):
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
    if toy:
        dataset["train"] = tokens2seqs(train_data[:500])
    else:
        dataset["train"] = tokens2seqs(train_data)

    dataset["dev"] = tokens2seqs(dev_data)
    dataset["test"] = tokens2seqs(test_data)

    if saved_data_pth != None:
      torch.save(dataset,saved_data_pth)

    return dataset,train_data,test_data


def make_conceptnet_joint_dataloader(finetune_setting, pretrain_setting, args):
    finetune_raw_dataset,train_data,test_data = make_conceptnet_finetune_dataset(finetune_setting.saved_data_pth, finetune_setting.raw_data_pth, finetune_setting.processed_data_pth, args.toy)
    pretrain_raw_dataset = make_conceptnet_pretrain_dataset(args, pretrain_setting)


    def make_tensor_dataset(dataset, max_len, finetune = True):
        input_ids = []
        output_ids = []
        output_masks = []
        loss_masks = []

        for input_id, output_id in dataset:
            input_id += [1] * (max_len - len(input_id))
            output_id += [1] * (max_len - len(output_id))
            input_id = torch.LongTensor(input_id).unsqueeze(0)
            output_id = torch.LongTensor(output_id).unsqueeze(0)
            output_mask = output_id != 1
            if finetune:
                loss_mask = torch.ones((1,1))
            else:
                loss_mask = torch.zeros((1,1))

            input_ids.append(input_id)
            output_ids.append(output_id)
            output_masks.append(output_mask)
            loss_masks.append(loss_mask)

        # input_ids = torch.cat(input_ids, dim = 0)
        # output_ids = torch.cat(output_ids, dim = 0)
        # output_masks = torch.cat(output_masks, dim = 0)
        # loss_masks = torch.cat(loss_masks, dim = 0)

        dataset = (input_ids, output_ids, output_masks, loss_masks)

        return dataset

    joint_train_finetune = make_tensor_dataset(finetune_raw_dataset["train"], max_len = args.max_len, finetune = True)
    joint_train_pretrain = make_tensor_dataset(pretrain_raw_dataset["train"][:len(finetune_raw_dataset["train"])], max_len = args.max_len, finetune = False)
    joint_train = [torch.cat(joint_train_finetune[i] + joint_train_pretrain[i], dim = 0) for i in range(len(joint_train_finetune))]
    # return joint_train, train_data, test_data
    joint_train = torch.utils.data.TensorDataset(joint_train[0],joint_train[1],joint_train[2],joint_train[3])

    train_dataloader  = torch.utils.data.DataLoader(
        joint_train,  # The training samples.
        sampler=torch.utils.data.RandomSampler(joint_train),  # Select batches randomly
        batch_size=args.train_batch_size  # Trains with this batch size.
        )


    joint_dev_finetune = make_tensor_dataset(finetune_raw_dataset["dev"], max_len = args.max_len, finetune = True)
    joint_dev_pretrain = make_tensor_dataset(pretrain_raw_dataset["test"][:len(finetune_raw_dataset["dev"])], max_len = args.max_len, finetune = False)

    joint_dev = [torch.cat(joint_dev_finetune[i] + joint_dev_pretrain[i], dim = 0) for i in range(len(joint_dev_finetune))]

    joint_dev = torch.utils.data.TensorDataset(joint_dev[0],joint_dev[1],joint_dev[2],joint_dev[3])

    dev_dataloader  = torch.utils.data.DataLoader(
        joint_dev,  # The training samples.
        sampler=torch.utils.data.RandomSampler(joint_dev),  # Select batches randomly
        batch_size=args.eval_batch_size  # Trains with this batch size.
        )

    
    test_dataset = make_tensor_dataset(finetune_raw_dataset["test"], max_len = 50, finetune = True)
    test_dataset = [torch.cat(test_dataset[i],dim = 0) for i in range(len(test_dataset))]
    test_dataset = torch.utils.data.TensorDataset(test_dataset[0],test_dataset[1],test_dataset[2],test_dataset[3])

    test_dataloader  = torch.utils.data.DataLoader(
        test_dataset,  # The training samples.
        sampler=torch.utils.data.RandomSampler(test_dataset),  # Select batches randomly
        batch_size=args.eval_batch_size  # Trains with this batch size.
    )


    return {"train":train_dataloader,"dev":dev_dataloader,"test":test_dataloader}, train_data, test_data
    


    
    

    
    
    

    

def process_pretrain_path2path_raw_data(raw_data_pth,processed_data_pth = None):
    raw_data = torch.load(raw_data_pth)     #list of tuples
    print(raw_data_pth)
    # raw_data = pickle.load(open(raw_data_pth,"rb"))
    processed_dataset = []
    for tuple_path in tqdm.tqdm(raw_data):
        input_lst = mege_to_sequence(tuple_path[0])
        output_lst = mege_to_sequence(tuple_path[1])
        input_ids = lst2ids(input_lst)
        output_ids = lst2ids(output_lst)
        processed_dataset.append((input_ids,output_ids))

    if processed_data_pth != None:
        torch.save(processed_dataset,processed_data_pth)

    return processed_dataset

def  process_pretrain_path_raw_data(setting,raw_data_pth,saved_data_pth):
    # raw_data = torch.load(raw_data_pth)     #list of tuples
    print(raw_data_pth)
    raw_data = pickle.load(open(raw_data_pth,"rb"))
    if setting.corruption == "none":
        processed_dataset = []
        for tuple_path in tqdm.tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            input_ids = lst2ids(input_lst)
            output_ids = lst2ids(output_lst)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset

    if setting.corruption == "deletion":
        ratio = float(setting.ratio)
        processed_dataset = []
        for tuple_path in tqdm.tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_deletion(output_ids,ratio)
            
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset
        
        
    if setting.corruption == "masking":
        ratio = float(setting.ratio)
        processed_dataset = []
        for tuple_path in tqdm.tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_masking(output_ids,ratio)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset

    if setting.corruption == "infilling":
        processed_dataset = []
        for tuple_path in tqdm.tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_infilling(output_ids,setting.num_of_span,setting.lamda)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset




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

    



