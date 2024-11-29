import os
import argparse
os.environ["TRANSFORMERS_OFFLINE"]= 'True'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import warnings
import json

warnings.filterwarnings("ignore")

import argparse

from itertools import chain
import nltk
from nlgeval import compute_metrics

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
import nltk
from datasets import load_metric

def get_dist(file):
    res = {}
    itr = 0
    for sentence in file:
        res[itr] = nltk.word_tokenize(sentence)
        itr += 1
    uni_grams = []
    bi_grams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        uni_grams += ugs
        bi_grams += bgs
        ma_dist1 += len(set(ugs)) / float(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / float(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(uni_grams)) / float(len(uni_grams) + 1e-16)
    mi_dist2 = len(set(bi_grams)) / float(len(bi_grams) + 1e-16)
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def evaluateModel(predict_file, truth_file):
    f = open(predict_file, 'r')
    g = open(truth_file, 'r')
    preds = []
    refs = []
    generation = f.readline().replace('\n', '')
    source = g.readline().replace('\n', '')
    preds.append(generation)

    refs.append(source)

    while source and generation:
        generation = f.readline().replace('\n', '')
        source = g.readline().replace('\n', '')
        preds.append(generation)
        refs.append(source)
    preds = preds[0:len(preds) - 1]
    refs = refs[0:len(refs) - 1]
    _, _, d1, d2, _ = get_dist(preds)

    metrics_dict = compute_metrics(hypothesis=predict_file, references=[truth_file], no_skipthoughts=True,
                                no_glove=True)
    
    predict_file_name=os.path.splitext(os.path.split(predict_file)[-1])[0]
    metrics_dict['d1'] = d1
    metrics_dict['d2'] = d2
    metrics_dict['name'] = predict_file_name
    
    if not os.path.exists('./eval_results'):
        os.mkdir('./eval_results')
    if os.path.exists(f'./eval_results/{predict_file_name}.json'):
        with open(f'./eval_results/{predict_file_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    data.append(metrics_dict)
    print(metrics_dict)
    with open(f'./eval_results/{predict_file_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return metrics_dict

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--predict_file',required=True)
    parser.add_argument('-t','--truth_file',required=True)
    args=parser.parse_args()

    evaluateModel(args.predict_file, args.truth_file)
