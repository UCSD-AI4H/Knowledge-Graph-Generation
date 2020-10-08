from atomic_data import make_atomic_finetune_dataset, make_atomic_pretrain_dataset
from conceptnet_data import make_conceptnet_finetune_dataset, make_conceptnet_pretrain_dataset

def make_finetune_dataset(args,finetune_setting):
    if args.dataset_type == "conceptnet":
        dataset = make_conceptnet_finetune_dataset(saved_data_pth = finetune_setting.saved_data_pth,
                                                   raw_data_pth = finetune_setting.raw_data_pth, 
                                                   processed_data_pth = finetune_setting.processed_data_pth,
                                                   toy=args.toy)
    
    if args.dataset_type == "atomic":
        dataset = make_atomic_finetune_dataset(saved_data_pth = finetune_setting.saved_data_pth,
                                                   raw_data_pth = finetune_setting.raw_data_pth, 
                                                   processed_data_pth = finetune_setting.processed_data_pth,
                                                   toy=args.toy)

    return dataset



def make_pretrain_dataset(args,pretrain_setting):
    if args.dataset_type == "conceptnet":
        dataset = make_conceptnet_pretrain_dataset(args,pretrain_setting)
    
    if args.dataset_type == "atomic":
        dataset = make_atomic_pretrain_dataset(args,pretrain_setting)

    return dataset