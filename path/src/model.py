from transformers import BartForConditionalGeneration,BartConfig
import torch
from atomic_data import tokenizer
from copy import deepcopy


def make_model(args):
    if args.dataset_type == "conceptnet":
        model = make_conceptnet_model(args.model_pth)
    
    if args.dataset_type == "atomic":
        model = make_atomic_model(args.model_pth)

    return model


def make_conceptnet_model(model_pth = None):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    if model_pth != None and model_pth != "None":
        print("model loaded from {}".format(model_pth))
        model_state_dict = torch.load(model_pth)["model"]
        model.load_state_dict(model_state_dict)
    
    return model


def make_atomic_model(model_pth = None):
    config = BartConfig.from_pretrained("facebook/bart-large")
    config.vocab_size = len(tokenizer.encoder)

    model = BartForConditionalGeneration(config)

    if model_pth != None and model_pth != "None":
        print("model loaded from {}".format(model_pth))
        model_state_dict = torch.load(model_pth)["model"]
        model.load_state_dict(model_state_dict)
        return model


    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    new_state_dict = {}
    new_state_dict = deepcopy(model.state_dict())
    bart_state_dict = bart_model.state_dict()

    for key in bart_state_dict:
        if key in ["model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]:
            new_state_dict[key][:50265,:] = deepcopy(bart_state_dict[key])
            continue
        if key == "final_logits_bias":
            continue
    
        new_state_dict[key] = deepcopy(bart_state_dict[key])
  
    model.load_state_dict(new_state_dict)

    return model