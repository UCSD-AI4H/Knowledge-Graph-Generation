import sys
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="conceptnet",
                    choices=["conceptnet","atomic"])
parser.add_argument("--epoch_num", type=str, default="10")
parser.add_argument("--gpt2_model_type", type=str, default="small",
                    choices=["small","medium"])
parser.add_argument("--method", type=str, default="pretrain",
                    choices=["ckg","pretrain"])
parser.add_argument("--mode", type=str, default="train", 
                    choices=["train","generate"])
parser.add_argument("--lr", type=str, default="1e-5")
parser.add_argument("--num_gradients_accumulation", type=str, default="4")
parser.add_argument("--batch_size", type=str, default="32")
parser.add_argument("--train_data", type=str, default="data/train_data_rgcn_conceptnet.pkl")
parser.add_argument("--test_data", type=str, default="data/test_data_rgcn_conceptnet.pkl")
parser.add_argument("--model_file", type=str, default="models/new_model.pkl")
parser.add_argument("--batch_size", type=str, default="32")
parser.add_argument("--restore", action="store_true")



args = parser.parse_args()


if args.method == "pretrain":
    if args.dataset == "conceptnet":
        from pretrain_conceptnet import main 
    if args.dataset == "atomic":
        from pretrain_atomic import main
    main()
if args.method == "ckg":
    print("not implemented")
















