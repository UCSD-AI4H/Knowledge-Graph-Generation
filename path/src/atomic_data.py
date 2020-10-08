from transformers import BartTokenizer
import tqdm
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
categories = []
categories += ["oEffect"]
categories += ["oReact"]
categories += ["oWant"]
categories += ["xAttr"]
categories += ["xEffect"]
categories += ["xIntent"]
categories += ["xNeed"]
categories += ["xReact"]
categories += ["xWant"]

special = []
special += ["<{}>".format(cat) for cat in categories]

start_id = tokenizer.encoder["<s>"]
end_id = tokenizer.encoder["</s>"]
mask_id = tokenizer.encoder["<mask>"]
pad_id = tokenizer.encoder["<pad>"]


for special_token in special:
    tokenizer.decoder[len(tokenizer.encoder)] = special_token
    tokenizer.encoder[special_token] = len(tokenizer.encoder)


def make_dataset(raw_data):
  dataset = []
  for prefix, cat, suffix in tqdm.tqdm(raw_data):
    if suffix == "none":
      continue
    new_prefix = prefix.replace("___","<mask>")
    input_id = tokenizer.encode(new_prefix)[1:-1]
    input_id += [tokenizer.encoder[cat]]
    input_id = [start_id] + input_id + [end_id]
    # print(suffix)
    output_id = tokenizer.encode(suffix)

    dataset.append((input_id, output_id))

  return dataset

def make_atomic_finetune_dataset(saved_data_pth = None,raw_data_pth = None, processed_data_pth = None, toy = False):
    raw_data = torch.load(raw_data_pth)

    if processed_data_pth != None:
        full_dataset = torch.load(raw_data_pth)

    else:
        if toy:
            train_dataset = make_dataset(raw_data["train"]["total"][:500])
            dev_dataset = make_dataset(raw_data["dev"]["total"][:500])
            test_dataset = make_dataset(raw_data["test"]["total"][:500])
        else:
            train_dataset = make_dataset(raw_data["train"]["total"])
            dev_dataset = make_dataset(raw_data["dev"]["total"])
            test_dataset = make_dataset(raw_data["test"]["total"])

        full_dataset = {"train":train_dataset, "dev":dev_dataset,"test":test_dataset}

    if saved_data_pth != None:
        torch.save(full_dataset)

    return full_dataset

def make_atomic_pretrain_dataset(args,pretrain_setting):
    dataset = {}        #TODO:

    return dataset
