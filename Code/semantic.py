import argparse
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class senmanticSimilarity():
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def cal(self, seq1, seq2):
        results = torch.tensor([])
        for sen1, sen2 in tqdm(zip(seq1, seq2)):
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
    simCalculator = senmanticSimilarity()
    sim = simCalculator.process(PREDICT_FILE, TRUTH_FILE)
    print(sim)