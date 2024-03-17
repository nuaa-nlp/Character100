import os

import torch
torch.manual_seed(42)
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import re

from dataset import DiscriminatorDataset
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("/data/xwang/llama2/llama2-7b-chat-hf", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("/data/xwang/llama2/llama2-7b-chat-hf", device_map="auto", trust_remote_code=True, load_in_8bit=True, torch_dtype=torch.float16)

model = PeftModel.from_pretrained(model, "/data/xwang/llama2/llama-2-7b-chat-discriminator-lora/checkpoint-2150", torch_dtype=torch.float16)

def generate_prompt(speech):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Based on the input, determine whose style of speaking this sentence is. Just give names, don't output other information. The outputs should be in the following format: <name>.
### Input:
{speech}
### Response:
"""

def processPrompt(prompt):
    result = re.findall(r'### Response:\s+(.*)',prompt)
    
    return result[0]

def get_data(filePath):
    with open(filePath, 'r') as f:
        data = f.readlines()
    names = []
    contexts = []
    questions = []
    answers = []
    for line in data:
        name, context, question, answer = line.strip().split('\t')
        name = name.replace('_',' ')
        names.append(name)
        contexts.append(context)
        questions.append(question)
        answers.append(answer)
    return names, contexts, questions, answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str)
    parser.add_argument('--maxnewtoken', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--topp', type=float, default=0.9)
    parser.add_argument('--batchsize', type=int, default=2)
    args = parser.parse_args()

    args = parser.parse_args()
    names, contexts, questions, answers = get_data('./dataset/test_combined_ver2.txt')
    testset = DiscriminatorDataset(names, contexts, questions, answers, args.maxnewtoken)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False)
    with open(f'./discriminator_results/result_{args.temperature}_{args.topp}_2150.txt', 'w') as f1:
        with torch.no_grad():
            for batch in tqdm(testloader):
                prompts, truth = batch
                prompts = [generate_prompt(i) for i in prompts]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to('cuda')
                # print('generating..')
                predicts = model.generate(**inputs, max_new_tokens=args.maxnewtoken,do_sample=True,repetition_penalty=1.1, top_p = args.topp, temperature = args.temperature,  num_return_sequences = 5)
                predicts = predicts.reshape((5,-1))
                pre_list = []
                for i in range(predicts.shape[0]):
                    pre_list.append(tokenizer.decode(predicts[i], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                
                pre_list = [processPrompt(i) for i in pre_list]
                predicts = [pre.replace('<','').replace('>','') for pre in pre_list]
                for pre, name in zip(predicts, truth):
                    f1.write(','.join(predicts)+'\t'+name+'\n')