import os

import torch
torch.manual_seed(42)
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import re

from chacater_100_dataset import DiscriminatorEvalDataset
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm

class StyleDiscriminator():
    def __init__(self, mode, llama2_7b_chat_path,discriminator_ckpt_path,eval_data_path,testset_path,maxnewtoken,temperature,topp,batchsize):
        with open(testset_path, 'r') as f:
            data = f.readlines()
        self.names = []
        for line in data:
            name = line.strip().split('\t')[0]
            name = name.replace('_',' ')
            self.names.append(name)
        self.maxnewtoken=maxnewtoken
        self.temperature=temperature
        self.topp=topp
        self.batchsize=batchsize
        self.mode=mode
        self.eval_data_path=eval_data_path
        if self.mode=='predict':
            self.tokenizer = AutoTokenizer.from_pretrained(llama2_7b_chat_path, use_fast=False, trust_remote_code=True)
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(llama2_7b_chat_path, device_map="auto", trust_remote_code=True, load_in_8bit=True, torch_dtype=torch.float16)

            self.model = PeftModel.from_pretrained(self.model, discriminator_ckpt_path, torch_dtype=torch.float16)
        self.get_dataloader(eval_data_path)


    def generate_prompt(self,speech):
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

    def get_dataloader(self, eval_data_path):
        with open(eval_data_path, 'r') as f:
            data = f.readlines()
        predicts = [item.strip() for item in data]
        
        evalset = DiscriminatorEvalDataset(predicts)
        self.evalloader = DataLoader(evalset, batch_size=self.batchsize, shuffle=False)
    
    def predict(self, fp):
        predicts_all = []
        with torch.no_grad():
            for batch in tqdm(self.evalloader):
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
        print(predicts_all)
        for item in predicts_all:
            fp.write(','.join(item)+'\n')
        return predicts_all
    
    def load_predict(self, filepath):
        with open(filepath) as f:
            data=f.readlines()
        return [item.strip().split(',') for item in data]

    def eval(self, predicts, truths):
        at = [[] for _ in range(5)]
        fail = []
        for index, predict in tqdm(enumerate(predicts), total=len(predicts)):
            predicts_list = [self.process_special_names(i) for i in predict]
            truth = self.process_special_names(truths[index])
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

        return [float(hitAt1)/float(len(predicts)), float(hitAt3)/float(len(predicts)), float(hitAt5)/float(len(predicts))]
    
    def discriminate(self):
        if not os.path.exists('./discriminator_results'):
            os.mkdir('./discriminator_results')
        output_filename=os.path.split(self.eval_data_path)[-1]
        predict_file=os.path.join('./discriminator_results',output_filename)
        if self.mode=='predict':
            with open(predict_file, 'w') as f1:
                with torch.no_grad():
                    self.predict(f1)
            predicts=self.load_predict(predict_file)
            results = self.eval(predicts, self.names)
            return results
        elif self.mode=='eval':
            predicts=self.load_predict(predict_file)
            results = self.eval(predicts, self.names)
            return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1','--llama2_7b_chat_path')
    parser.add_argument('-p2','--discriminator_ckpt_path',default='Discriminator_checkpoint/')
    parser.add_argument('-p3','--testset_path',type=str,default='Data/test.txt')
    parser.add_argument('-m','--mode',choices=['predict','eval'],required=True)
    parser.add_argument('-p4','--eval_data_path',type=str)
    parser.add_argument('--maxnewtoken', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--topp', type=float, default=0.9)
    parser.add_argument('--batchsize', type=int, default=4)
    args = parser.parse_args()

    discriminator=StyleDiscriminator(args.mode,args.llama2_7b_chat_path,args.discriminator_ckpt_path,args.eval_data_path,args.testset_path,args.maxnewtoken,args.temperature,args.topp,args.batchsize)
    results=discriminator.discriminate()
    print(results)