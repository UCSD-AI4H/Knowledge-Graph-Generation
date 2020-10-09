import sys
import os
import argparse


parser = argparse.ArgumentParser()


parser.add_argument("--dataset", type=str, default="conceptnet",
                    choices=["conceptnet","atomic"])
parser.add_argument("--method", type=str, default="pretrain",
                    choices=["ckg","pretrain"])
parser.add_argument("--epoch_num", type=str, default="10")
parser.add_argument("--lr", type=str, default="1e-5")
parser.add_argument("--num_gradients_accumulation", type=str, default="4")
parser.add_argument("--batch_size", type=str, default="32")
parser.add_argument("--train_data", type=str, default="data/train_data_rgcn_conceptnet")
parser.add_argument("--test_data", type=str, default="data/test_data_rgcn_conceptnet")
parser.add_argument("--model_file", type=str, default="models/new_model")
parser.add_argument("--log_file", type=str, default="log/new_log")
parser.add_argument("--batch_size", type=str, default="32")
parser.add_argument("--load_model", action="store_true")

args = parser.parse_args()

if args.method == "pretrain":
    if args.dataset == "conceptnet":
        from pretrain_conceptnet import main 
    if args.dataset == "atomic":
        from pretrain_atomic import main
    main()
if args.method == "ckg":
    if args.dataset == "conceptnet":
        from train_conceptnet import main 
    if args.dataset == "atomic":
        from train_atomic import main
    main()




