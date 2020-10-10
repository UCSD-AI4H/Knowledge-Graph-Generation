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

def lm_loss(logits, labels, lable_masks, reduction = "mean"):

    bsz = logits.shape[0]
    out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
    out = F.log_softmax(out)
    target = labels[:, 1:].contiguous().reshape(-1)

    loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
    loss = (loss * lable_masks[:,1:].float()).sum(1)
    length = lable_masks[:,1:].float().sum(1)
    loss = loss / length

    if reduction == "mean":
        loss = loss.sum()/bsz

    return loss

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = net
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, w_optim, Likelihood, step, batch_size):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        dataIndex = len(trn_y)+step*batch_size
        
        input_id, input_mask = trn_X
        output_id, output_mask = trn_y
           
        # forward
        logits = self.v_net(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]
        
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = lm_loss(logits, output_id, output_mask, reduction="none")
        # print(first.size())
        # print(second.size())
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        loss = lossup/lossdiv
        
#         loss = torch.dot(torch.sigmoid(Likelihood[step*batch_size:dataIndex]), ignore_crit(logits, trn_y))/(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        
        # compute gradient of train loss towards likelihhod
        loss.backward()

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw in zip(self.net.parameters(), self.v_net.parameters()):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                
                if w.grad is not None:
                    vw.copy_(w - args.lr * (m + w.grad + self.w_weight_decay*w))


    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, w_optim, Likelihood, Likelihood_optim, step, batch_size):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # crit = nn.CrossEntropyLoss().cuda()
        
        xi = 0.01
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, w_optim, Likelihood, step, batch_size)
        
        
        vinput_id, vinput_mask = val_X
        voutput_id, voutput_mask = val_y
        # calc val prediction
        logits = self.v_net(input_ids = vinput_id, decoder_input_ids = voutput_id, labels = voutput_id)[1]
        # calc unrolled validation loss
        loss = lm_loss(logits, voutput_id, voutput_mask, reduction='mean')# L_val(w`)
        
        # compute gradient of validation loss towards weights
        v_weights = tuple(self.v_net.parameters())
        # some weights not used return none
        
        dw = []
        for w in v_weights:  
            if w.requires_grad:
                dw.append(torch.autograd.grad(loss, w, allow_unused=True, retain_graph=True))
            else:
                dw.append(None)
        hessian = self.compute_hessian(dw, trn_X, trn_y, Likelihood, batch_size, step)

        
        Likelihood_optim.zero_grad()
        # update final gradient = - xi*hessian
#         with torch.no_grad():
#             for likelihood, h in zip(Likelihood, hessian):
#                 print(len(hessian))
#                 likelihood.grad = - xi*h
        with torch.no_grad():
            Likelihood.grad = - xi*hessian[0]         
        Likelihood_optim.step()
        return Likelihood, Likelihood_optim, loss

    def compute_hessian(self, dw, trn_X, trn_y, Likelihood, batch_size, step):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
                
        norm = torch.cat([w[0].view(-1) for w in dw if ((w != None) and (w[0] != None))]).norm()
        
        eps = 0.01 / norm
        
        input_id, input_mask = trn_X
        output_id, output_mask = trn_y
        
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d!= None and d[0] != None:
                    pp = eps * d[0]
                    p += eps * d[0]
        
        
        # forward & calc loss
        dataIndex = len(input_id)+step*batch_size 
        # forward
        logits = self.net(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = lm_loss(logits, output_id, output_mask, reduction="none")
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        loss = lossup/lossdiv
        
        
        dalpha_pos = torch.autograd.grad(loss, Likelihood) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d != None and d[0] != None:
                    p -= 2. * eps * d[0]
        # forward
        logits = self.net(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]
        # sigmoid loss
        first = torch.sigmoid(Likelihood[step*batch_size:dataIndex])
        second = lm_loss(logits, output_id, output_mask, reduction="none")
        lossup = torch.dot(first, second)
        lossdiv =(torch.sigmoid(Likelihood[step*batch_size:dataIndex]).sum())
        loss = lossup/lossdiv


        dalpha_neg = torch.autograd.grad(loss, Likelihood) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                if d != None and d[0] != None:
                    p += eps * d[0]

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian




def train(model,dataset,test_data,train_data,optimizer,log_path,gen_path,best_model_pth,batch_size = 64, num_accumulation = 2, steps = 50000, epoch_num = 10):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    dev_dataset = dataset["dev"]
    iter_num = len(train_dataset) // batch_size 
    best_ppl = 10000000
    best_score = 0
    step_count = 0
    total_loss = []

    architect = Architect(model,w_momentum=0.9,w_weight_decay = 3e-4)
    Likelihood = torch.nn.Parameter(torch.ones(len(train_dataset)).cuda(),requires_grad=True).cuda()
    Likelihood_optim = torch.optim.Adam({Likelihood}, 0.1, betas=(0.5, 0.999))

    bar = tqdm.tqdm(total=steps)
    bar.update(0)

    logs = {}
    begin_eval = False
    dev_iter = 0
    total_dev_iter = len(dev_dataset) // batch_size

    while step_count < steps:
      model.train()
      epoch_loss = 0
      optimizer.zero_grad()
      random.shuffle(train_dataset)
      for iter in range(iter_num):
        input_id, input_mask, output_id, output_mask = gen_batched_data(batch_size, iter, train_dataset)
        vinput_id, vinput_mask, voutput_id, voutput_mask = gen_batched_data(batch_size, dev_iter, dev_dataset)

        trn_x = (input_id,input_mask)
        trn_y = (output_id,output_mask)
        val_x = (vinput_id,vinput_mask)
        val_y = (voutput_id,voutput_mask)

        bsz = input_id.shape[0]
        

        architect.unrolled_backward(trn_x,trn_y,val_x,val_y,optimizer,Likelihood,Likelihood_optim,iter,bsz)


        logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

        loss = lm_loss(logits,output_id,output_mask,reduction="mean")

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
          test_perplexity, log = do_eval(model, dev_dataset, test_dataset, test_data, train_data, gen_path, step_count, steps)
          log["macro loss"] = sum(total_loss) / len(total_loss)
          logs[str(step_count)] = log
          torch.save(logs, log_path + "/{}-{}.pkl".format(step_count, steps))
          begin_eval = False
          if test_perplexity < best_ppl:
            best_ppl = test_perplexity
            #save model
            torch.save({"model":model.state_dict(), "opt":optimizer}, open(best_model_pth,"wb"))
            print("best ppl model saved")
        
        dev_iter = (dev_iter + 1) % total_dev_iter
 

def do_eval(model, dev_dataset, test_dataset, test_data, train_data, gen_path, step_count, steps):
#   eval_perplexity = eval_model(model,dev_dataset)
  test_perplexity = eval_model(model,test_dataset)

  gen_name = gen_path + "/iter{}-{}.txt".format(step_count, steps)
  log_info = bart_evaluate.evaluate_generation_dataset(test_dataset,test_data,train_data,model,gen_name=gen_name)
  log_info["Test Perplexity"] = test_perplexity
#   log_info["Eval Perplexity"] = eval_perplexity
  log_info["step"] = step_count

  print("================Step %d===================="%(step_count))
  print("Saving gens to {}".format(gen_name))
#   print("Eval Perplexity: %f"%(eval_perplexity))
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
  input_ids = gen_batched_data(sample_num, 0, test_dataset)[0]
  output_ids = model.generate(input_ids=input_ids, max_length=20,do_sample=False)
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



    


  
  


