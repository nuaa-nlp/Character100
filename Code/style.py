import torch
import os

from dataset import DiscriminatorEvalDataset
torch.manual_seed(42)
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse

from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm

class StyleDiscriminator:
    def __init__(self, test_file_path, batchsize=4) -> None:
        with open(test_file_path, 'r') as f:
            data = f.readlines()
        self.names = []
        for line in data:
            name = line.strip().split('\t')[0]
            name = name.replace('_',' ')
            self.names.append(name)
        self.tokenizer = AutoTokenizer.from_pretrained("/llama2-7b-chat-hf", use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained("/llama2-7b-chat-hf", device_map="auto", trust_remote_code=True, load_in_8bit=True, torch_dtype=torch.float16)

        self.model = PeftModel.from_pretrained(self.model, "/Discriminator_checkpoint", torch_dtype=torch.float16)
        self.batchsize = batchsize

    def predict(self, predict_file, batchsize=4):
        with open(predict_file, 'r') as f:
            predict_sentences = f.readlines()
        predict_dataset = DiscriminatorEvalDataset(predict_sentences)
        predict_loader = DataLoader(predict_dataset, batch_size=batchsize, shuffle=False)
        predicts_all = []
        with torch.no_grad():
            for batch in tqdm(predict_loader):
                predicts = batch
                prompts = [self.generate_prompt(i) for i in predicts]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to('cuda')
                predicts = self.model.generate(**inputs, max_new_tokens=20,do_sample=True,repetition_penalty=1.1, top_p = 0.9, temperature = 0.9,  num_return_sequences = 5)
                pre_list =[]

                for i in range(predicts.shape[0]):
                    now_predict = self.tokenizer.decode(predicts[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    pre_list.append(now_predict)
                
                
                pre_list = [self.processPrompt(i).replace('<','').replace('>','') for i in pre_list]
                pre_list = [pre_list[i:i+5] for i in range(0, len(pre_list), 5)]
                predicts_all+=pre_list
        
        return predicts_all
    
    def generate_prompt(self, speech):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Based on the input, determine whose style of speaking this sentence is. Just give names, don't output other information. The outputs should be in the following format: <name>.
### Input:
{speech}
### Response:
"""
    def processPrompt(self, prompt):
        result = re.findall(r'### Response:\s+(.*)',prompt)

        return result[0]    
    def process_special_names(self, string:str):
        string = string.replace('Diana, Princess of Wales', 'Diana; Princess of Wales')
        string = string.replace('Prince Philip, Duke of Edinburgh', 'Prince Philip; Duke of Edinburgh')
        string = string.replace('Princess Margaret, Countess of Snowdon', 'Princess Margaret; Countess of Snowdon')
        string = string.replace('Margaret, Countess of Snowdon', 'Princess Margaret; Countess of Snowdon')
        string = string.replace('Meghan, Duchess of Sussex', 'Meghan; Duchess of Sussex')
        string = string.replace('William, Prince of Wales', 'William; Prince of Wales')
        string = string.replace('Prince William, Duke of Cambridge', 'Prince William; Duke of Cambridge')
        string = string.replace('Catherine, Princess of Wales', 'Catherine; Princess of Wales')
        string = string.replace('Victoria, Queen of the United Kingdom', 'Queen Victoria')
        
        return string
    
    def eval(self, predicts):
        at = [[] for _ in range(5)]
        fail = []
        for index, predict in enumerate(predicts):
            predicts_list = [self.process_special_names(i) for i in predict]
            truth = self.process_special_names(self.names[index])
            hit = False
            for j, pre in enumerate(predicts_list):
                if pre == truth:
                    hit = True
                    at[j].append(str(index))
                    break
            if not hit:
                fail.append(str(index))
        
        hitAt1 = len(at[0])
        hitAt3 = len(at[0]) + len(at[1]) + len(at[2])
        hitAt5 = hitAt3 + len(at[3]) + len(at[4])

        return [float(hitAt1)/float(len(self.names)), float(hitAt3)/float(len(self.names)), float(hitAt5)/float(len(self.names))]
    
    def discriminate(self, predict_file):
        predicts = self.predict(predict_file, self.batchsize)
        results = self.eval(predicts)
        return [str(i) for i in results]

if __name__ == '__main__':
    style = StyleDiscriminator('/Data/test.txt', args.batchsize)
    results = style.discriminate(PREDICT_FILE)
    print(results)