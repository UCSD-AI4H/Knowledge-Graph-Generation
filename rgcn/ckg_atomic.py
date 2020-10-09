import torch
import pandas as pd
import pickle
import ckbc_demo.demo_bilinear as demo_bilinear
import sys
import os
import argparse
import nltk
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu


from model.pretrain_model_atomic.R_GCN_GPT2


parser = argparse.ArgumentParser()

parser.add_argument("--test_data", type=str, default="data/test_data_ckg_atomic")
parser.add_argument("--model_file", type=str, default="models/new_model")
parser.add_argument("--output_file", type=str, default="log/evaluate_result")
parser.add_argument("--load_model", action="store_true")


args = parser.parse_args()


def get_s_r(testset):
    return [item[0] + '\t' + item[1] for item in testset]

def generate_objects(model,testset):
    batch_size = 20
    iter = len(testset) // batch_size
    st, ed = 0, 0
    results = []
    for iteration in tqdm.tqdm_notebook(range(iter)):
        st = ed
        ed += batch_size

        batch_data = np.array(testset[st:ed])
        batch = {}
        batch['sen_ids'] = torch.LongTensor(batch_data[:,0].tolist()).to(device)
        batch['sr_ids'] = torch.LongTensor(batch_data[:,1].tolist()).to(device)
        batch['sen_mask'] = torch.LongTensor(batch_data[:,2].tolist()).to(device)
        batch['loss_mask'] = torch.LongTensor(batch_data[:,3].tolist()).to(device)
        batch['sr_mask'] = torch.LongTensor(batch_data[:,4].tolist()).to(device)
        sentence = model.generate_bert(batch)
    
        for i in range(sentence.shape[0]):
            sen = sentence[i].tolist()
            # print('sen:',sen)
            if 50256 in sen:
                idx = sen.index(50256)
            else:
                idx = -1
            results.append(tokenizer_gpt2.decode(sen[:idx]).strip())

    return results

def prepare_for_evaluation(s_r, obj, gen_name):
  # print('s_r:',len(s_r))
  # print('obj:',len(obj))
  assert(len(s_r) == len(obj))
  
  lines = []
  for i in range(len(s_r)):
    s,r = s_r[i].split("\t")
    o = obj[i]
    line = "{}\t{}\t{}\t1\n".format(s,r,o)
    lines.append(line)
    
  print("Generated on {}".format(gen_name))
  with open(gen_name, "w") as fout:
    fout.writelines(lines)


def novelty_evaluation(gen_name, triples):
    train_sro = set([(s,r,o) for s,r,o in triples])
    train_o = set([o for s,r,o in triples])
    
    with open(gen_name, "r") as fin:
        lines = fin.readlines()

    generate_sro = []
    generate_o = []

    for line in lines:
        s,r,o,_ = line.split("\t")
        generate_sro.append((s,r,o))
        generate_o.append(o)

    # generate_sro = set(generate_sro)
    # generate_o = set(generate_o)

    N_sro = 0
    N_o = 0

    for sro in generate_sro:
        if sro not in train_sro:
            N_sro += 1

    for o in generate_o:
        if o not in train_o:
            N_o += 1


    print("N_sro:",N_sro / len(generate_sro) * 100)
    print("N_o:",N_o / len(generate_sro) * 100)


    return N_sro / len(generate_sro) * 100, N_o / len(generate_sro) * 100



def evaluate_generation(test_dataset, triples, test_trp, model, gen_name):
    log_info = {}

    obj = generate_objects(model,test_dataset)
    s_r = get_s_r(test_trp)
    prepare_for_evaluation(s_r,obj,gen_name)

    # Do Novelty Evaluation
    n_sro,n_o = novelty_evaluation(gen_name,triples)

    log_info["N_sro"] = n_sro
    log_info["N_o"] = n_o

    # Calculate Score
    output_triples = []
    with open(gen_name,'r') as fin:
        lines = fin.readlines()
        for line in lines:
            h,r,t,_ = line.split('\t')
            output_triples.append((h,r,t))

    h_past = None
    r_past = None

    temp_gens = []
    test_triples = test_trp
    gen_past = None
    gen = None
    for i in range(len(test_triples)):
        h,r,_ = test_triples[i]
        if h != h_past or r != r_past:
            if gen != None:
                gen['refs'] = refs
                temp_gens.append(gen)
                
            h_past = h
            r_past = r
            gen = {}a
            gen['event'] = h + ' ' + r
            gen['beams'] = []
            t = output_triples[i][-1]
            refs = []
            
        refs.append(test_triples[i][-1].split())
        gen['beams'].append(t.split()) 

    n = 2
    # Set score
    weights = [1/n] * n

    def score(hyp, refs):
        return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

    # Compute BLEU
    total_bl = 0
    total_count = 0
    for gen in tqdm.tqdm(temp_gens):
        event = gen["event"]
        list_of_gens = gen['beams']
        list_of_refs = gen['refs']

        if sum([i == ["none"] for i in list_of_refs]) / len(list_of_refs) > 1/3:
            continue

        example_bl = []

        for clean_gen in list_of_gens:
    #         print(list_of_refs)
            a = score(clean_gen, list_of_refs)
            example_bl.append(a)
            
        total_bl += sum(example_bl)
        total_count += len(example_bl)

    print("{}: \t {}".format(score, total_bl / total_count))

    return log_info

if __name__ == "__main__":
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

    atomic_data = pickle.load(open('data/atomic_data','rb'))
    triple_list = [triple for triple in atomic_data['train']['total'] if triple[-1] != 'none']
    test_triples = [triple for triple in atomic_data['test']['total'] if triple[-1] != 'none']
    test_triples = test_triples[:72900]
    
    test_data_name = args.test_data
    output_file = args.output_file
    model_file = args.model_file
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

    if args.load_model == True:
        model.load_state_dict(torch.load(model_file))
        evaluate_generation(test_data, triple_list, test_triples, model, output_file)
    else:
        print("Need to load trained model")




