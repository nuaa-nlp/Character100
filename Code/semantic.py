import argparse
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class semanticSimilarity():
    def __init__(self, model_path) -> None:
        self.model = SentenceTransformer(model_path)
    
    def cal(self, seq1, seq2):
        results = torch.tensor([])
        for sen1, sen2 in tqdm(zip(seq1, seq2), total=len(seq1)):
            embedding_1= self.model.encode(sen1, convert_to_tensor=True)
            embedding_2 = self.model.encode(sen2, convert_to_tensor=True)

            results = torch.cat((results,util.pytorch_cos_sim(embedding_1, embedding_2).cpu()))

        return results.squeeze().mean().item()
    
    def process(self, predict_file, truth_file):
        with open(predict_file, 'r') as f1:
            predicts = f1.readlines()
        with open(truth_file, 'r') as f2:
            truths = f2.readlines()
        sim = self.cal(predicts, truths)
        return sim

#Compute embedding for both lists

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--predict_file',required=True)
    parser.add_argument('-t','--truth_file',required=True)
    parser.add_argument('-m','--model_path',required=True)
    args=parser.parse_args()
    
    simCalculator = semanticSimilarity(args.model_path)
    sim = simCalculator.process(args.predict_file, args.truth_file)
    print(sim)