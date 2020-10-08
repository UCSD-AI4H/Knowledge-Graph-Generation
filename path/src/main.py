import torch
import argparse
from model import make_model
from data import make_pretrain_dataset, make_finetune_dataset
from conceptnet_data import make_conceptnet_joint_dataloader
import pretrain
# import finetune
import joint_training
from opt import OpenAIAdam
from collections import namedtuple
import utils
import os
import finetune

def main(args,pretrain_setting,finetune_setting):
    # Build Model and push model to GPU
    device = torch.device('cuda')
    if args.do_pretrain or args.do_joint:
        model = make_model(args)
    else:
        model = make_model(args)
    
    model = model.to(device)

    #Build dataset
    if args.do_joint:
        data_loaders,train_data,test_data = make_conceptnet_joint_dataloader(finetune_setting,pretrain_setting,args)

    if args.do_pretrain:
        pretrain_dataset = make_pretrain_dataset(args,pretrain_setting)
    if args.do_finetune:
        finetune_dataset, train_data, test_data = make_finetune_dataset(args,finetune_setting)

    #Start_training_process
    if args.do_joint:
        if args.resume:
          check_point = torch.load(args.resume_path)
          optimizer = check_point["optimizer"]
          model.load_state_dict(check_point["model"])

          joint_training.train(model,
                              data_loaders = data_loaders,
                              test_data = test_data,
                              train_data = train_data,
                              optimizer = optimizer,
                              log_path = args.log_path, 
                              gen_path = finetune_setting.gen_pth,
                              trade_off = args.trade_off,
                              best_model_pth = args.best_model_pth,
                              epoch_num = args.epoch_num,
                              num_accumulation = args.num_accumulation)

        else:
          num_train_optimization_steps = args.steps

          optimizer = OpenAIAdam(model.parameters(),
                                  lr=1e-5,
                                  schedule='warmup_linear',
                                  warmup=0.002,
                                  t_total=num_train_optimization_steps,
                                  b1=0.9,
                                  b2=0.999,
                                  e=1e-08,
                                  l2=0.01,
                                  vector_l2=True,
                                  max_grad_norm=args.clip)

          joint_training.train(model,
                              data_loaders = data_loaders,
                              test_data = test_data,
                              train_data = train_data,
                              optimizer = optimizer,
                              log_path = args.log_path, 
                              gen_path = args.gen_pth,
                              trade_off = args.trade_off,
                              best_model_pth = args.best_model_pth,
                              epoch_num = args.epoch_num,
                              steps = args.steps,
                              num_accumulation = args.num_accumulation)
    if args.do_pretrain:
        num_train_optimization_steps = args.steps
        optimizer = OpenAIAdam(model.parameters(),
                                lr=1e-5,
                                schedule='warmup_linear',
                                warmup=0.002,
                                t_total=num_train_optimization_steps,
                                b1=0.9,
                                b2=0.999,
                                e=1e-08,
                                l2=0.01,
                                vector_l2=True,
                                max_grad_norm=args.clip)
        pretrain.train(model,
                       dataset = pretrain_dataset,
                       optimizer = optimizer,
                       log_path = args.log_path,
                       best_model_pth = args.best_model_pth,
                       batch_size=args.train_batch_size,
                       num_accumulation=args.num_accumulation,
                       steps = args.steps,
                       epoch_num=args.epoch_num)

    if args.do_finetune:
        if args.do_pretrain:
          model = make_model(args.load_model_pth)
        # num_train_optimization_steps = len(finetune_dataset["train"]) * args.epoch_num // args.train_batch_size // args.num_accumulation
        num_train_optimization_steps = args.steps
        print_train_information(args, num_train_optimization_steps)
        optimizer = OpenAIAdam(model.parameters(),
                                lr=1e-5,
                                schedule='warmup_linear',
                                warmup=0.002,
                                t_total=num_train_optimization_steps,
                                b1=0.9,
                                b2=0.999,
                                e=1e-08,
                                l2=0.01,
                                vector_l2=True,
                                max_grad_norm=args.clip)
        finetune.train(model,
                       dataset = finetune_dataset,
                       test_data = test_data,
                       train_data = train_data,
                       optimizer = optimizer,
                       log_path = args.log_path,
                       gen_path = args.gen_pth,
                       best_model_pth = args.best_model_pth,
                       batch_size=args.train_batch_size,
                       num_accumulation=args.num_accumulation,
                       steps = args.steps,
                       epoch_num=args.epoch_num)

def print_train_information(args,num_train_optimization_steps):
  print("================Training Start From Here================")
  print("log path: ", args.log_path)
  print("best model path: ", args.best_model_pth)
  print("Total Steps: ",num_train_optimization_steps)
  print("Epoch Num: ", args.epoch_num)
  print("Num Accumulation: ", args.num_accumulation)
  print("Batch Size: ", args.train_batch_size)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--do_pretrain',action='store_true')
    parser.add_argument('--do_finetune',action='store_true')
    parser.add_argument('--do_joint',action='store_true')
    parser.add_argument('--dataset_type',type=str,default='conceptnet')
    parser.add_argument('--pretrain_type',type=str,default='path')
    parser.add_argument('--pretrain_config_pth',type=str,default='pretrain_config.json')
    parser.add_argument('--finetune_config_pth',type=str,default='finetune_config.json')
    parser.add_argument('--train_batch_size',type=int, default= 8)
    parser.add_argument('--eval_batch_size',type=int, default= 32)
    parser.add_argument('--toy',action='store_true')
    parser.add_argument('--max_len',type=int,default = 100)
    parser.add_argument("--resume", action = 'store_true')
    parser.add_argument("--resume_path", type=str, default = "finetune_models/joint_2.pkl")
    parser.add_argument("--log_path", type = str, default = "finetune_log/test.pkl")
    parser.add_argument("--best_model_pth", type = str, default = "finetune_models/test.pkl")
    parser.add_argument("--load_model_pth", type = str, default = "pretrain_models/test.pkl")
    parser.add_argument("--trade_off",type = float, default = 0.7)
    parser.add_argument("--epoch_num",type = int, default = 10)
    parser.add_argument("--steps", type = int, default = 50000)
    parser.add_argument("--num_accumulation", type = int, default = 2)
    parser.add_argument("--gen_pth",type = str, default = "gen/")
    parser.add_argument("--clip", type = float, default = 1)
    parser.add_argument("--overwrite", action = "store_true")

    args = parser.parse_args()


    if not args.overwrite:
      if args.do_finetune or args.do_joint:
        assert(False == os.path.exists(args.gen_pth))
        assert(False == os.path.exists(args.log_path))
        assert(False == os.path.exists(args.best_model_pth))

        os.mkdir(args.gen_pth)
        os.mkdir(args.log_path)

      if args.do_pretrain:
        assert(False == os.path.exists(args.log_path))
        assert(False == os.path.exists(args.best_model_pth))

        os.mkdir(args.log_path)



    pretrain_setting = utils.load_config(args.pretrain_config_pth,pretrain = True)
    finetune_setting = utils.load_config(args.finetune_config_pth,pretrain = False)
    print(args)
    main(args,pretrain_setting,finetune_setting)

