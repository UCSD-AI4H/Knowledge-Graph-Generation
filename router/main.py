import torch
import argparse
from model import make_model,Customized_Bart
from data import make_pretrain_dataset, make_finetune_dataset
from pytorch_pretrained_bert import OpenAIAdam
import pretrain
import finetune
from opt import OpenAIAdam1
from collections import namedtuple
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import time
import pickle
from torch.utils.data import TensorDataset, DataLoader
import sys
import os





def main(args, finetune_setting):
    #Build Model and push model to GPU
    device = torch.device('cuda:0')
    if args.do_pretrain:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        epochs=args.epoch_num
        num_gradients_accumulation=args.num_gradients_accumulation
        batch_size=args.batch_size
        gpu_id=0
        lr=args.lr
        MAX_ENCODER_LEN = 30
        MAX_DECODER_LEN = 200

        train_data = torch.load(args.train_data) 
        train_data = train_data[0:54336]   #in this test, due to time limitation, only test 54336 pieces of data.

        train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=16)
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model = Customized_Bart(bart_model).to(device)

        optimizer, criterion = pretrain.get_optimizer(model, epochs, len(train_data), batch_size,
                                            num_gradients_accumulation=num_gradients_accumulation, lr=lr)

        pretrain.train(args, model, device, train_dataloader, batch_size, epochs, optimizer, criterion, num_gradients_accumulation,MAX_ENCODER_LEN)
    else:
        model = torch.load(args.model_file)
        new_state_dict = {}
        for key in list(model.keys()):
          new_key = key
          if key != 'final_logits_bias':
            new_key = "model." + key
          new_state_dict[new_key] = model[key]
        from transformers import BartForConditionalGeneration
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        base_model.load_state_dict(new_state_dict)
        torch.save(base_model.state_dict(),"Bart_pretrain_10_1010.pkl")
        model = make_model(finetune_setting.load_model_pth)
    
    model = model.to(device)
    #model = torch.load("Bart_pretrain.pkl")
    #Build dataset
    
  
    if args.do_finetune:
        finetune_dataset, train_data, test_data = make_finetune_dataset(saved_data_pth = finetune_setting.saved_data_pth,
                                            raw_data_pth = finetune_setting.raw_data_pth, 
                                            processed_data_pth = finetune_setting.processed_data_pth)
    #print(finetune_dataset)
   

    if args.do_finetune:
        if args.do_pretrain:
          model = make_model(finetune_setting.load_model_pth)
        if args.ignore:
          import finetune as finetune
        else:
          import finetune_ignore as finetune
        num_train_optimization_steps = len(finetune_dataset["train"]) * finetune_setting.epoch_num // finetune_setting.batch_size // finetune_setting.num_accumulation
        optimizer = OpenAIAdam1(model.parameters(),
                                lr=1e-5,
                                schedule='warmup_linear',
                                warmup=0.002,
                                t_total=num_train_optimization_steps,
                                b1=0.9,
                                b2=0.999,
                                e=1e-08,
                                l2=0.01,
                                vector_l2=True,
                                max_grad_norm=1)
        finetune.train(model,
                       dataset = finetune_dataset,
                       test_data = test_data,
                       train_data = train_data,
                       optimizer = optimizer,
                       log_path = finetune_setting.log_pth,
                       gen_path = finetune_setting.gen_pth,
                       best_model_pth = finetune_setting.best_model_pth,
                       batch_size=finetune_setting.batch_size,
                       num_accumulation=finetune_setting.num_accumulation,
                       epoch_num=finetune_setting.epoch_num)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--do_pretrain',action='store_true')
    parser.add_argument('--do_finetune',action='store_true')
    #parser.add_argument('--pretrain_config_pth',type=str,default='pretrain_config.json')
    parser.add_argument('--finetune_config_pth',type=str,default='finetune_config.json')
    parser.add_argument('--ignore',action='store_true')

    parser.add_argument("--dataset", type=str, default="conceptnet")
    parser.add_argument("--method", type=str, default="pretrain")
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_gradients_accumulation", type=int, default= 2)
    parser.add_argument("--batch_size", type=int, default= 16)
    parser.add_argument("--train_data", type=str, default="datas/train_data_10_inputs2.pth")
    parser.add_argument("--model_file", type=str, default="models/Bart_parameter_10.pkl")
    parser.add_argument("--log_file", type=str, default="log/new_log")
    parser.add_argument("--load_model", action="store_true")

    args = parser.parse_args()


   # pretrain_setting = utils.load_config(args.pretrain_config_pth,pretrain = True)
    finetune_setting = utils.load_config(args.finetune_config_pth,pretrain = False)

    main(args, finetune_setting)

