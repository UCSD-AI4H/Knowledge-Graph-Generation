import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from bart_evaluate import combine_into_words
import pickle
from data import tokenizer
import bart_evaluate
import random

device = torch.device("cuda")
eval_batch_size = 32


def train(model,dataset,optimizer,log_path,best_model_pth,batch_size = 64, num_accumulation = 2, steps = 50000, epoch_num = 10):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    iter_num = len(train_dataset) // batch_size 
    best_ppl = 10000000
    best_score = 0
    step_count = 0
    total_loss = []

    bar = tqdm.tqdm(total=steps)
    bar.update(0)

    logs = {}
    begin_eval = False

    while step_count < steps:
      model.train()
      epoch_loss = 0
      optimizer.zero_grad()
      random.shuffle(train_dataset)
      for iter in range(iter_num):
        input_id, input_mask, output_id, output_mask = gen_batched_data(batch_size, iter, train_dataset)
        bsz = input_id.shape[0]
        logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

        out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
        out = F.log_softmax(out)
        target = output_id[:, 1:].contiguous().reshape(-1)
    
        loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
        loss = (loss * output_mask[:,1:].float()).sum(1)
        length = output_mask[:,1:].float().sum(1)
        # length = output_mask.float().sum(1)
        loss = (loss/length).sum()/bsz

        loss.backward()

        epoch_loss += loss.item()
        total_loss.append(loss.item())

        if (iter + 1) % num_accumulation == 0:
          optimizer.step()
          optimizer.zero_grad()
          step_count += 1
          if step_count >= steps:
            break

          if (step_count % 500) == 0:
            begin_eval = True
          bar.update(1)

        if begin_eval:
          test_perplexity, log = do_eval(model, test_dataset, step_count, steps)
          log["macro loss"] = sum(total_loss) / len(total_loss)
          print("macro loss:", log["macro loss"])
          sample(model,test_dataset,8)
          logs[str(step_count)] = log
          torch.save(logs, log_path + "/{}-{}.pkl".format(step_count, steps))
          begin_eval = False
          if test_perplexity < best_ppl:
            best_ppl = test_perplexity
            #save model
            torch.save({"model":model.state_dict(), "opt":optimizer}, open(best_model_pth,"wb"))
            print("best ppl model saved")
 

def do_eval(model, test_dataset, step_count, steps):
  test_perplexity = eval_model(model,test_dataset)

  log_info = {}
  log_info["Test Perplexity"] = test_perplexity
  log_info["step"] = step_count

  print("================Step %d===================="%(step_count))
  print("Test Perplexity: %f"%(test_perplexity))

  return test_perplexity, log_info

  
    


  
  # break

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





def sample(model, test_dataset,sample_num = 5):
  input_ids = gen_batched_data(sample_num, 0, test_dataset)[0]
  output_ids = model.generate(input_ids=input_ids, max_length=60,do_sample=False)
  for i in range(output_ids.shape[0]): #  3 output sequences were generated
    print('Generated {}: {} {}'.format(i, tokenizer.decode(input_ids[i], skip_special_tokens=True), tokenizer.decode(output_ids[i], skip_special_tokens=True)))


def eval_model(model, eval_dataset):
  eval_iter_num = len(eval_dataset) // eval_batch_size
  model.eval()
  perplexity = 0
  for iter in range(eval_iter_num):
    with torch.no_grad():
      input_id, input_mask, output_id, output_mask = gen_batched_data(eval_batch_size, iter, eval_dataset)
      bsz = input_id.shape[0]
      logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

      out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
      out = F.log_softmax(out)
      # print(out.shape)
      target = output_id[:, 1:].contiguous().reshape(-1)
      # print(target.shape)

      loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
      loss = (loss * output_mask[:,1:].float()).sum(1)
      length = output_mask[:,1:].float().sum(1)
      loss = (loss/length).sum()/bsz

      perplexity += loss.item()

  return np.exp(perplexity / eval_iter_num)



    


  
  


