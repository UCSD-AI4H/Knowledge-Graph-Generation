from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import numpy as np
import ckbc_demo.demo_bilinear as demo_bilinear

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
device = torch.device("cuda")

combine_into_words = {
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

batch_size = 32


def gen_batched_data(batch_size, iter, dataset, PAD_IDX = 1):
  st = iter * batch_size
  ed = min([(iter+1) * batch_size, len(dataset)])
  batched_data = dataset[st:ed]

  max_input_len = max([len(data[0]) for data in batched_data])
  max_output_len = max([len(data[1]) for data in batched_data])

  batched_input_id = []
  batched_output_id = []

  for input_id, output_id in batched_data:
    input_id += [PAD_IDX] * (max_input_len - len(input_id))
    output_id += [PAD_IDX] * (max_output_len - len(output_id))
    batched_input_id.append(input_id)
    batched_output_id.append(output_id)

  batched_input_id = torch.LongTensor(batched_input_id).to(device)
  batched_output_id = torch.LongTensor(batched_output_id).to(device)
  batched_input_mask = batched_input_id != PAD_IDX
  batched_output_mask = batched_output_id != PAD_IDX

  return batched_input_id, batched_input_mask, batched_output_id, batched_output_mask


def gen_batched_data_setting2(batch_size, iter, dataset, PAD_IDX = 1):
  st = iter * batch_size
  ed = min([(iter+1) * batch_size, len(dataset)])
  batched_data = dataset[st:ed]

  max_input_len = max([len(data[0]) for data in batched_data])
  max_output_len = max([len(data[1]) for data in batched_data])

  batched_input_id = []
  batched_output_id = []

  for input_id, output_id, _ in batched_data:
    input_id += [PAD_IDX] * (max_input_len - len(input_id))
    output_id += [PAD_IDX] * (max_output_len - len(output_id))
    batched_input_id.append(input_id)
    batched_output_id.append(output_id)

  batched_input_id = torch.LongTensor(batched_input_id).to(device)
  batched_output_id = torch.LongTensor(batched_output_id).to(device)
  batched_input_mask = batched_input_id != PAD_IDX
  batched_output_mask = batched_output_id != PAD_IDX

  return batched_input_id, batched_input_mask, batched_output_id, batched_output_mask



def generate_objects(test_dataset,model):
  outputs = []
  s_r = []
  for batch in test_dataset:
    input_ids = batch[0].to(device)
    # output_ids = model.generate(input_ids=input_ids, max_length=50,do_sample=True,num_beams = 10)
    output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=False)
    # output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True, temperature = 0.8)
    for i in range(output_ids.shape[0]): #  3 output sequences were generated
      outputs.append(tokenizer.decode(output_ids[i], skip_special_tokens=True))
      s,r = parse_sr(input_ids[i])
      s_r.append(s + '\t' + r)
  return s_r,outputs

def generate_objects_dataset(test_dataset,model):
  outputs = []
  s_r = []
  eval_batch_size = 32
  iter_num = len(test_dataset) // eval_batch_size + 1
  for iter in range(iter_num):
    input_ids = gen_batched_data(eval_batch_size, iter, test_dataset)[0].to(device)
    # output_ids = model.generate(input_ids=input_ids, max_length=50,do_sample=True,num_beams = 10)
    output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=False)
    # output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True, temperature = 0.8)
    for i in range(output_ids.shape[0]): #  3 output sequences were generated
      outputs.append(tokenizer.decode(output_ids[i], skip_special_tokens=True))
  return outputs

def generate_objects_dataset_setting2(test_dataset,model):
  outputs = []
  s_r = []
  eval_batch_size = 32
  iter_num = len(test_dataset) // eval_batch_size + 1
  for iter in range(iter_num):
    input_ids = gen_batched_data_setting2(eval_batch_size, iter, test_dataset)[0].to(device)
    # output_ids = model.generate(input_ids=input_ids, max_length=50,do_sample=True,num_beams = 10)
    output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=False)
    # output_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True, temperature = 0.8)
    for i in range(output_ids.shape[0]): #  3 output sequences were generated
      gen_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
      outputs.append(parse_obj(gen_text))
  return outputs

def parse_sr(input_id):
  s_r_str = tokenizer.decode(input_id, skip_special_tokens = True)
  for key in combine_into_words:
    if key in s_r_str:
      s,_ = s_r_str.split(key, 1)
      s = s.strip()
      r = combine_into_words[key]
      break
  return s,r


def parse_obj(input_str):
  for key in combine_into_words:
    if key in input_str:
      _,obj = input_str.split(key,1)
      obj = obj.strip()
      break
  return obj

  
def get_s_r(raw_data):
    return [s + "\t" + combine_into_words[r] for s,r,_,_ in raw_data]

def prepare_for_evaluation(s_r, obj, gen_name):
  assert(len(s_r) == len(obj))
  lines = []
  for i in range(len(s_r)):
    s,r = s_r[i].split("\t")
    o = obj[i]
    line = "{}\t{}\t{}\t1\n".format(s,r,o)
    lines.append(line)
    
  print("Generated on {}".format(gen_name))
  with open(gen_name, "w") as fout:
    fout.writelines(lines)


def novelty_evaluation(gen_name, raw_train_data):
    train_sro = set([(s,combine_into_words[r],o) for s,r,o,_ in raw_train_data])
    train_o = set([o for _,_,o,_ in raw_train_data])
    
    with open(gen_name, "r") as fin:
        lines = fin.readlines()

    generate_sro = []
    generate_o = []

    for line in lines:
        s,r,o,_ = line.split("\t")
        generate_sro.append((s,r,o))
        generate_o.append(o)

    # generate_sro = set(generate_sro)
    # generate_o = set(generate_o)

    N_sro = 0
    N_o = 0

    for sro in generate_sro:
        if sro not in train_sro:
            N_sro += 1

    for o in generate_o:
        if o not in train_o:
            N_o += 1


    print("N_sro:",N_sro / len(generate_sro) * 100)
    print("N_o:",N_o / len(generate_sro) * 100)


    return N_sro / len(generate_sro) * 100, N_o / len(generate_sro) * 100



def evaluate_generation(test_dataset, raw_test_data, raw_train_data, model, gen_name, thresh = 0.5):
    log_info = {}

    s_r,obj = generate_objects(test_dataset,model)
    prepare_for_evaluation(s_r,obj,gen_name)

    #Do Novelty Evaluation
    n_sro,n_o = novelty_evaluation(gen_name,raw_train_data)

    log_info["N_sro"] = n_sro
    log_info["N_o"] = n_o

    #Use Bilinear AVG for Evaluation
    results = demo_bilinear.run(gen_name, flip_r_e1=True)

    new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}
    num_examples = 1.0 * len(results)
    positive = sum(np.array(new_results["1"]) > thresh)
    # accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
    #             len([i for i in new_results["0"] if i < args.thresh])) / num_examples
    accuracy = positive / num_examples

    log_info["score"] = accuracy * 100
    
    print("Score @ {}: {}".format(thresh, accuracy*100))

    return log_info

    

def evaluate_generation_dataset(test_dataset, raw_test_data, raw_train_data, model, gen_name, thresh = 0.5):
    log_info = {}

    obj = generate_objects_dataset(test_dataset,model)
    s_r = get_s_r(raw_test_data)
    prepare_for_evaluation(s_r,obj,gen_name)

    #Do Novelty Evaluation
    n_sro,n_o = novelty_evaluation(gen_name,raw_train_data)

    log_info["N_sro"] = n_sro
    log_info["N_o"] = n_o

    #Use Bilinear AVG for Evaluation
    results = demo_bilinear.run(gen_name, flip_r_e1=True)

    new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}
    num_examples = 1.0 * len(results)
    positive = sum(np.array(new_results["1"]) > thresh)
    # accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
    #             len([i for i in new_results["0"] if i < args.thresh])) / num_examples
    accuracy = positive / num_examples

    log_info["score"] = accuracy * 100
    
    print("Score @ {}: {}".format(thresh, accuracy*100))

    return log_info


def evaluate_generation_dataset_setting2(test_dataset, raw_test_data, raw_train_data, model, gen_name, thresh = 0.5):
    log_info = {}

    obj = generate_objects_dataset_setting2(test_dataset,model)
    s_r = get_s_r(raw_test_data)
    prepare_for_evaluation(s_r,obj,gen_name)

    #Do Novelty Evaluation
    n_sro,n_o = novelty_evaluation(gen_name,raw_train_data)

    log_info["N_sro"] = n_sro
    log_info["N_o"] = n_o

    #Use Bilinear AVG for Evaluation
    results = demo_bilinear.run(gen_name, flip_r_e1=True)

    new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}
    num_examples = 1.0 * len(results)
    positive = sum(np.array(new_results["1"]) > thresh)
    # accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
    #             len([i for i in new_results["0"] if i < args.thresh])) / num_examples
    accuracy = positive / num_examples

    log_info["score"] = accuracy * 100
    
    print("Score @ {}: {}".format(thresh, accuracy*100))

    return log_info










    