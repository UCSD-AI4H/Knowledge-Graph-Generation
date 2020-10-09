import torch
import pandas as pd
import pickle
import ckbc_demo.demo_bilinear as demo_bilinear
import sys
import os
import argparse

from model.pretrain_model_conceptnet.R_GCN_GPT2


parser = argparse.ArgumentParser()

parser.add_argument("--test_data", type=str, default="data/test_data_ckg_conceptnet")
parser.add_argument("--model_file", type=str, default="models/new_model")
parser.add_argument("--output_file", type=str, default="log/evaluate_result")
parser.add_argument("--load_model", action="store_true")


args = parser.parse_args()

def get_s_r(testset):
    return [item[0] + '\t' + item[2] for item in testset]

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
        sentence = model.generate_ckg(batch)
    
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
    train_sro = set([(s,o,r) for s,r,o in triples])
    train_o = set([o for s,o,r in triples])
    
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



def evaluate_generation(test_dataset, triples, test_trp, model, gen_name, thresh = 0.5):
    log_info = {}

    obj = generate_objects(model,test_dataset)
    s_r = get_s_r(test_trp)
    prepare_for_evaluation(s_r,obj,gen_name)

    #Do Novelty Evaluation
    n_sro,n_o = novelty_evaluation(gen_name,triples)

    log_info["N_sro"] = n_sro
    log_info["N_o"] = n_o

    #Use Bilinear AVG for Evaluation
    results = demo_bilinear.run(gen_name, flip_r_e1=True)

    new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}
    num_examples = 1.0 * len(results)
    positive = sum(np.array(new_results["1"]) > thresh)
    # accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
    #             len([i for i in new_results["0"] if i < args.thresh])) / num_examples
    accuracy = positive / num_examples

    log_info["score"] = accuracy * 100
    
    print("Score @ {}: {}".format(thresh, accuracy*100))
    # for i in range(len(obj)):
    #     print(s_r[i] + '\t' + obj[i])

    return log_info

if __name__ == "__main__":
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

    triple_list = pickle.load(open('data/conceptnet_triples','rb'))
    test_triples = pickle.load(open('data/conceptnet_test_triples','rb'))
    
    test_data_name = args.test_data
    output_file = args.output_file
    model_file = args.model_file
    test_data = pickle.load(open(test_data_name,'rb'))
    
    # load model
    model = R_GCN_GPT2().to(device)

    if args.load_model == True:
        model.load_state_dict(torch.load(model_file))
        evaluate_generation(test_data, triple_list, test_triples, model, output_file)
    else:
        print("Need to load trained model")




