import torch
import torch.nn as nn
import numpy as np
import times
import torch.optim as optim
import pickle
import copy,math
import tqdm
import dgl
from transformers import GPT2LMHeadModel,GPT2Tokenizer,GPT2Model
import dgl.function as fn
from pytorch_pretrained_bert import OpenAIAdam


from model.pretrain_model_atomic.R_GCN_GPT2
from main import args


def get_optimizer(model, epochs, train_data_length, batch_size, num_gradients_accumulation=4, lr=1e-5):
    # optimizer
    num_train_optimization_steps = train_data_length * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=lr,
                        warmup=0.01,
                        max_grad_norm=1.0,
                        weight_decay=0.01,
                        t_total=num_train_optimization_steps)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    return optimizer, criterion


def train(model, train_data, test_data, epochs, optimizer, criterion, num_gradients_accumulation, model_file):
    start = time.time()
    update_count = 0
    # print('test_data:',len(test_data))
    iter = len(train_data) // batch_size
    current_ppl = 10000
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        st, ed = 0, 0
        np.random.shuffle(train_data)
        
        for iteration in tqdm.tqdm(range(iter)):
            st = ed
            ed += batch_size
            
            # optimizer.zero_grad()

            batch = {'g':[],'names':[],'edge_types':[],'edge_norms':[],'path':[],'path_mask':[]}
            for piece in train_data[st:ed]:
                batch['g'].append(piece[0].to(device))
                batch['names'].append(piece[1])
                batch['edge_types'].append(piece[2])
                batch['edge_norms'].append(piece[3])
                batch['path'].append(piece[4])
                batch['path_mask'].append(piece[5])


            batch['path'] = torch.LongTensor(batch['path']).to(device)
            batch['path_mask'] = torch.LongTensor(batch['path_mask']).to(device)
            
            logits = model(batch)
            logits = logits[0]
            # print('logits.shape:',logits[0].shape)
            out = logits[:, 1:-1].contiguous() # mask graph logits
            target = batch['path'][:, 1:].contiguous()
            # print('target:',target.shape)
            # print('target:',target)
            target_mask = batch['path_mask'][:, 1:].contiguous()

            out = out.reshape(-1, out.shape[-1])
            target = target.reshape(-1)
            target_mask = target_mask.reshape(-1)
            # print('target:',target.shape)
            # print('out:',out.shape)
            loss = criterion(out, target)
            loss = loss.masked_fill_(mask=(0==target_mask),value=0)
            
            loss = torch.sum(loss) / torch.sum(target_mask)
            torch.cuda.empty_cache()
            loss.backward(retain_graph=True)
            
            
            # optimizer.step()
            losses += loss.item()
            times += 1
            # print('Batch Loss:',loss.item())
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        
        start = end
        
        #------------------------validate------------------------
        model.eval()

        perplexity = 0
        batch_count = 0
        # print('test_data:',len(test_data))
        iter_test = len(test_data) // batch_size
        st, ed = 0, 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for iteration in tqdm.tqdm(range(iter_test)):
                st = ed
                ed += batch_size

                batch = {'g':[],'names':[],'edge_types':[],'edge_norms':[],'path':[],'path_mask':[]}
                for piece in test_data[st:ed]:
                    batch['g'].append(piece[0].to(device))
                    batch['names'].append(piece[1])
                    batch['edge_types'].append(piece[2])
                    batch['edge_norms'].append(piece[3])
                    batch['path'].append(piece[4])
                    batch['path_mask'].append(piece[5])


                batch['path'] = torch.LongTensor(batch['path']).to(device)
                batch['path_mask'] = torch.LongTensor(batch['path_mask']).to(device)
                
                logits = model(batch)
                logits = logits[0]

                out = logits[:, 1:-1].contiguous()
                target = batch['path'][:, 1:].contiguous()
                target_mask = batch['path_mask'][:, 1:].contiguous()

                out = out.reshape(-1, out.shape[-1])
                target = target.reshape(-1)
                target_mask = target_mask.reshape(-1)
                # print('target:',target.shape)
                # print('out:',out.shape)
                loss = criterion(out, target)
                loss = loss.masked_fill_(mask=(0==target_mask),value=0)
                loss = torch.sum(loss) / torch.sum(target_mask)
                torch.cuda.empty_cache()
                perplexity += np.exp(loss.item())
                batch_count += 1

        ppl = perplexity / batch_count
        if ppl < current_ppl:
            current_ppl = ppl
            path = model_file
            torch.save(model.state_dict(), path)
            print("Save model with ppl", ppl)


        print(f'validate perplexity: {ppl}')



def main():
    epochs = int(args.epoch_num)
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_gradients_accumulation = int(args.num_gradients_accumulation)
    model_file = args.model_file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_data_name = args.train_data
    test_data_name = args.test_data

    train_data = pickle.load(open(train_data_name,'rb'))
    test_data = pickle.load(open(test_data_name,'rb'))

    # build model, need to change some layers due to added vocab_size for relations in ATOMIC
    model = R_GCN_GPT2().to(device)
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2') # for medium model, use 'gpt2-large'
    new_state_dict = copy.deepcopy(model.gpt2_model.state_dict())
    gpt2_state_dict = gpt2.state_dict()
    for key in gpt2_state_dict:
        if key in ['transformer.wte.weight','lm_head.weight']:
            new_state_dict[key][:50257,:] = copy.deepcopy(gpt2_state_dict[key])
            continue 
        new_state_dict[key] = copy.deepcopy(gpt2_state_dict[key])

    model.gpt2_model.load_state_dict(new_state_dict)

    if args.restore == True:
        model.load_state_dict(torch.load(model_file))

    optimizer, criterion = get_optimizer(model, epochs, len(train_data), batch_size,
                                         num_gradients_accumulation=num_gradients_accumulation, lr=lr)

    train(model, train_data, test_data, epochs, optimizer, criterion, num_gradients_accumulation, model_file)












