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
import time

device = torch.device("cuda")


def train(model,data_loaders,test_data,train_data,optimizer,log_path,gen_path,best_model_pth,trade_off = 0.9, batch_size = 64, num_accumulation = 2, steps = 100000, epoch_num = 10):
    train_dataloader = data_loaders["train"]  ##Used for Joint training of Pretraining and Finetuning
    test_dataloader = data_loaders["test"]   ##Used for Evaluate Finetuning only
    dev_dataloader = data_loaders["dev"]    ##Used for Evaluate Joint training 
    iter_num = len(train_dataloader)
    best_ppl = 10000000
    best_score = 0
    step_count = 0
    begin_eval = False

    bar = tqdm.tqdm(total=steps)
    bar.update(0)


    joint_loss_lst = []
    finetune_loss_lst = []
    pretrain_loss_lst = []
    log_info_lst = []
    # for epoch in range(epoch_num):
    while step_count < steps:
      model.train()
      optimizer.zero_grad()
    #   for iter in tqdm.tqdm(range(iter_num)):
      for i, batch_input in enumerate(train_dataloader):
        input_id, output_id, output_mask, loss_mask = [item.to(device) for item in batch_input]
        bsz = input_id.shape[0]
        logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

        out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
        out = F.log_softmax(out)
        target = output_id[:, 1:].contiguous().reshape(-1)
    
        loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
        loss = (loss * output_mask[:,1:].float()).sum(1)
        length = output_mask[:,1:].float().sum(1)
        loss = loss/length
        # print("loss",loss)
        loss_mask = loss_mask.squeeze(1)
        loss_attention = loss_mask.clone()
        loss_attention = loss_attention.masked_fill(loss_mask == 1, trade_off)
        loss_attention = loss_attention.masked_fill(loss_mask == 0, 1-trade_off)
        # print("loss_attention",loss_attention)

        with torch.no_grad():
            finetune_loss = (loss * loss_mask).sum() / loss_mask.sum() if (loss_mask.sum() != 0).item() else torch.zeros(1)
            pretrain_loss_mask = 1 - loss_mask
            pretrain_loss = (loss * pretrain_loss_mask).sum() / pretrain_loss_mask.sum() if (pretrain_loss_mask.sum() != 0).item() else torch.zeros(1)

        joint_loss = (loss * loss_attention).sum() / bsz

        joint_loss.backward()

        joint_loss_lst.append(joint_loss.item())
        finetune_loss_lst.append(finetune_loss.item())
        pretrain_loss_lst.append(pretrain_loss.item())

        if (i + 1) % num_accumulation == 0:
          optimizer.step()
          optimizer.zero_grad()
          step_count += 1
          bar.update(1)

          if step_count >= steps:
            break

          if (step_count % 500) == 0:
            begin_eval = True

        # time.sleep(0.05)
        if begin_eval:
          test_perplexity, log = do_eval(model, dev_dataloader, test_dataloader, test_data, train_data, gen_path, step_count, steps)
          log["macro joint loss"] = sum(joint_loss_lst) / len(joint_loss_lst)
          log["macro finetune loss"] = sum(finetune_loss_lst) / len(finetune_loss_lst)
          log["macro pretrain loss"] = sum(pretrain_loss_lst) / len(pretrain_loss_lst)
          print("Joint Loss:", log["macro joint loss"])
          print("Finetune Loss:", log["macro finetune loss"])
          print("Pretrain Loss", log["macro pretrain loss"])
          log_info_lst.append(log)
          torch.save(log_info_lst, log_path + "/{}-{}.pkl".format(step_count, steps))

          if test_perplexity < best_ppl:
            best_ppl = test_perplexity
            #save model
            torch.save({"model":model.state_dict(), "optimizer":optimizer}, open(best_model_pth,"wb"))
            print("best ppl model saved")

          begin_eval = False
          

        # if (i + 1) % 50 ==  0:
        #   pbar.update(1)
        #   print("Epoch %d\tStep: %d/%d\tJoint Loss: %f(%f)\tFinetune Loss: %f(%f)\tPretrain Loss: %f(%f)"%(epoch + 1, i + 1, len(train_dataloader), joint_loss.item(), sum(epoch_joint_loss)/len(epoch_joint_loss), finetune_loss.item(), sum(epoch_finetune_loss)/len(epoch_finetune_loss), pretrain_loss.item(), sum(epoch_pretrain_loss)/len(epoch_pretrain_loss)))

        
      # eval_perplexity = eval_joint_model(model,trade_off,dev_dataloader)
      # test_perplexity = eval_model(model,test_dataloader)

      # print("================Epoch %d===================="%(epoch))
      # print("Joint Training loss: %f"%(sum(epoch_joint_loss)/len(epoch_joint_loss)))
      # print("Finetune Training loss: %f"%(sum(epoch_finetune_loss)/len(epoch_finetune_loss)))
      # print("Pretrainig Training loss: %f"%(sum(epoch_pretrain_loss)/len(epoch_pretrain_loss)))

      # print("Eval Perplexity: %f"%(eval_perplexity))
      # print("Test Perplexity %f"%(test_perplexity))
      # sample(model,test_dataloader,10)

      # gen_path = gen_path + "{}epoch.txt".format(epoch)
      # log_info = bart_evaluate.evaluate_generation(test_dataloader,test_data,train_data,model,gen_name=gen_path)
      # log_info["Test Perplexity"] = test_perplexity
      # log_info["Eval Perplexity"] = eval_perplexity
      # log_info["Epoch"] = epoch + 1
      # log_info["step"] = (epoch + 1) * iter_num // 2
    #   log_to_file(log_path, log_info)
      # log_info_lst.append(log_info)
      
      # save_logs(joint_loss_lst, pretrain_loss_lst, finetune_loss_lst, log_info_lst, log_path)

      # if test_perplexity < best_ppl:
      #   best_ppl = test_perplexity
      #   #save model
      #   torch.save({"model":model.state_dict(), "optimizer":optimizer}, open(best_model_pth,"wb"))
      #   print("best ppl model saved")
    
def do_eval(model, dev_dataloader, test_dataloader, test_data, train_data, gen_path, step_count, steps):
  test_perplexity = eval_model(model,test_dataloader)

  print("================Step %d===================="%(step_count))
  print("Test Perplexity %f"%(test_perplexity))

  gen_name = gen_path + "/iter{}-{}.txt".format(step_count, steps)
  log_info = bart_evaluate.evaluate_generation(test_dataloader,test_data,train_data,model,gen_name=gen_name)
  log_info["Test Perplexity"] = test_perplexity
  log_info["step"] = step_count

  return test_perplexity, log_info

  # break

def save_logs(joint_loss_lst, pretrain_loss_lst, finetune_loss_lst, log_info, log_path):
    log = {"joint loss":joint_loss_lst, "pretrain_loss":pretrain_loss_lst, "finetune_loss": finetune_loss_lst, "evaluation_results":log_info}
    pickle.dump(log,open(log_path,'wb'))

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




        


  
def log_to_file(file_name, log_info):
  with open(file_name,"a") as fout:
    fout.write("==============Epoch {}=======================\n".format(log_info["Epoch"]))
    fout.write("Training Steps: {}\n".format(log_info["step"]))
    fout.write("Trainning Loss: {}\n".format(log_info["Trainning Loss"]))
    fout.write("Eval Perplexity: {}\n".format(log_info["Eval Perplexity"]))
    fout.write("Test Perplexity: {}\n".format(log_info["Test Perplexity"]))
    fout.write("N/T_sro: {}\n".format(log_info["N_sro"]))
    fout.write("N/T_o: {}\n".format(log_info["N_o"]))
    fout.write("Score: {}\n".format(log_info["score"]))



def sample(model, test_dataset,sample_num = 5):
  for batch in test_dataset:
    input_ids = batch[0][:sample_num].to(device)
    break

  output_ids = model.generate(input_ids=input_ids, max_length=20,do_sample=False)
  for i in range(output_ids.shape[0]): #  3 output sequences were generated
    print('Generated {}: {} {}'.format(i, tokenizer.decode(input_ids[i], skip_special_tokens=True), tokenizer.decode(output_ids[i], skip_special_tokens=True)))


def eval_model(model, eval_dataloader):
  model.eval()
  perplexity = 0
  for i, batch_input in enumerate(eval_dataloader):
    with torch.no_grad():
      input_id, output_id, output_mask, _ = [item.to(device) for item in batch_input]
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
      # length = output_mask.float().sum(1)
      loss = (loss/length).sum()/bsz

      perplexity += loss.item()

  return np.exp(perplexity / len(eval_dataloader))


def eval_joint_model(model,trade_off,eval_dataloader):
    model.eval()
    perplexity = 0
    for i, batch_input in enumerate(eval_dataloader):
        with torch.no_grad():
            input_id, output_id, output_mask, loss_mask = [item.to(device) for item in batch_input]
            bsz = input_id.shape[0]
            logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

            out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
            out = F.log_softmax(out)
            target = output_id[:, 1:].contiguous().reshape(-1)
        
            loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
            loss = (loss * output_mask[:,1:].float()).sum(1)
            length = output_mask[:,1:].float().sum(1)
            loss = loss/length
            loss_mask = loss_mask.squeeze(1)
            loss_attention = loss_mask
            loss_attention = loss_attention.masked_fill(loss_mask == 1, trade_off)
            loss_attention = loss_attention.masked_fill(loss_mask == 0, 1-trade_off)

            joint_loss = (loss * loss_attention).sum() / bsz

            perplexity += joint_loss.item()

    return np.exp(perplexity / len(eval_dataloader))




    


  
  


