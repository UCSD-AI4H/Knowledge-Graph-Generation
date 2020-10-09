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


from model.pretrain_model_conceptnet.R_GCN_GPT2
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


def train(model, train_data, test_data, epochs, optimizer, criterion, num_gradients_accumulation, log_file, model_file):
    start = time.time()
    update_count = 0
    cur_ppl = 10000
    infos = []
    pad_length = 50
    # print('test_data:',len(test_data))
    iter = len(train_data) // batch_size
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

            batch_data = np.array(train_data[st:ed])
            batch = {}
            batch['sen_ids'] = torch.LongTensor(batch_data[:,0].tolist()).to(device)
            batch['sr_ids'] = torch.LongTensor(batch_data[:,1].tolist()).to(device)
            batch['sen_mask'] = torch.LongTensor(batch_data[:,2].tolist()).to(device)
            batch['loss_mask'] = torch.LongTensor(batch_data[:,3].tolist()).to(device)
            batch['sr_mask'] = torch.LongTensor(batch_data[:,4].tolist()).to(device)
            
            logits = model.forward_ckg(batch)
            logits = logits[0]
            # print('logits.shape:',logits.shape)
            out = logits[:, :-1].contiguous() # mask graph logits
            target = batch['sen_ids'][:,1:].contiguous().to(device)
            # print('target:',target)
            target_mask = batch['loss_mask'][:,1:].contiguous().to(device)
            # print('out.shape:',out.shape) 
            # print('target.shape:',target_mask.shape)
            out = out.reshape(-1, out.shape[-1])
            target = target.reshape(-1)
            target_mask = target_mask.reshape(-1)
            
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
        loss_ = losses / times
        # path = 'GCN_GPT2_100k.pkl'
        # torch.save(model.state_dict(), path)
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

                batch_data = np.array(test_data[st:ed])
                batch = {}
                batch['sen_ids'] = torch.LongTensor(batch_data[:,0].tolist()).to(device)
                batch['sr_ids'] = torch.LongTensor(batch_data[:,1].tolist()).to(device)
                batch['sen_mask'] = torch.LongTensor(batch_data[:,2].tolist()).to(device)
                batch['loss_mask'] = torch.LongTensor(batch_data[:,3].tolist()).to(device)
                batch['sr_mask'] = torch.LongTensor(batch_data[:,4].tolist()).to(device)
                
                logits = model.forward_ckg(batch)
                logits = logits[0]
                # print('logits.shape:',logits.shape)
                out = logits[:, :-1].contiguous() # mask graph logits
                target = batch['sen_ids'][:,1:].contiguous().to(device)
                # print('target:',target)
                target_mask = batch['loss_mask'][:,1:].contiguous().to(device)
                # print('out.shape:',out.shape)
                # print('target.shape:',target_mask.shape)
                out = out.reshape(-1, out.shape[-1])
                target = target.reshape(-1)
                target_mask = target_mask.reshape(-1)
                
                loss = criterion(out, target)
                loss = loss.masked_fill_(mask=(0==target_mask),value=0)
                
                loss = torch.sum(loss) / torch.sum(target_mask)
                torch.cuda.empty_cache()
                perplexity += np.exp(loss.item())
                batch_count += 1

        ppl = perplexity / batch_count
        # evaluate the result
        # log_info = evaluate_generation(test_data,triple_list,test_triples,model,'output.txt')
        log_info['epoch'] = epoch + 1
        log_info['loss'] = loss_
        log_info['ppl'] = ppl
        infos.append(log_info)

        if ppl < cur_ppl:
            cur_ppl = ppl
            print('Store the model with ppl:',ppl)
            torch.save(model.state_dict(),'model_file')
        pickle.dump(infos,open('log_file','wb'))

        print(f'validate perplexity: {perplexity / batch_count}')


def main():
    epochs = int(args.epoch_num)
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_gradients_accumulation = int(args.num_gradients_accumulation)
    model_file = args.model_file
    log_file = args.log_file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_data_name = args.train_data
    test_data_name = args.test_data

    train_data = pickle.load(open(train_data_name,'rb'))
    test_data = pickle.load(open(test_data_name,'rb'))

    # load model
    model = R_GCN_GPT2().to(device)

    if args.load_model == True:
        model.load_state_dict(torch.load(model_file))

    optimizer, criterion = get_optimizer(model, epochs, len(train_data), batch_size,
                                         num_gradients_accumulation=num_gradients_accumulation, lr=lr)

    train(model, train_data, test_data, epochs, optimizer, criterion, num_gradients_accumulation, log_file, model_file)












