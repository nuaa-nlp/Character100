import os
import argparse
from background_knowledge import evaluateModel
from semantic import semanticSimilarity
from style import StyleDiscriminator
import torch
import datetime
import pandas as pd
import numpy as np

def eval_all(model_name,type,predict_path, truth_path, sbert_path,llama2_7b_chat_path,discriminator_ckpt_path,data_path,maxnewtoken,temperature,topp,batchsize):
    background_knowledge_metrics=evaluateModel(predict_path, truth_path)
    simCalculator = semanticSimilarity(sbert_path)
    sim = simCalculator.process(predict_path, truth_path)
    discriminator=StyleDiscriminator('predict',llama2_7b_chat_path,discriminator_ckpt_path,predict_path,data_path,maxnewtoken,temperature,topp,batchsize)
    discriminator_results=discriminator.discriminate()
    discriminator_results=np.average(np.array(discriminator_results)).item()

    if not os.path.exists('./overall_results'):
        os.mkdir('./overall_results')
    if not os.path.exists('./overall_results/result.csv'):
        df=pd.DataFrame([])
    else:
        df=pd.read_csv('./overall_results/result.csv')
    result={'model_name':model_name,'type':type,'time':datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),'Bleu_1':background_knowledge_metrics["Bleu_1"],'Bleu_2':background_knowledge_metrics["Bleu_2"],'Bleu_3':background_knowledge_metrics["Bleu_3"],'Bleu_4':background_knowledge_metrics["Bleu_4"],'semantic_similarity':sim,'style_similarity':discriminator_results}
    print(result)
    df=pd.concat([df,pd.DataFrame([result])],ignore_index=True)
    df.to_csv('./overall_results/result.csv',index=False)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', type=str, required=True)
    parser.add_argument('-t','--type',type=str,required=True)
    parser.add_argument('--root_path',type=str,default='results')
    parser.add_argument('--sbert_path',type=str,default='../all-MiniLM-L6-v2')
    parser.add_argument('-p1','--llama2_7b_chat_path')
    parser.add_argument('-p2','--discriminator_ckpt_path',type=str,default='Discriminator_checkpoint/')
    parser.add_argument('-p3','--data_path',type=str,default='Data/test.txt')
    parser.add_argument('--maxnewtoken', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--topp', type=float, default=0.9)
    parser.add_argument('--batchsize', type=int, default=32)
    args=parser.parse_args()
    
    predict_path=os.path.join(args.root_path,args.model_name,f'predict_{args.type}.txt')
    truth_path=os.path.join(args.root_path,args.model_name,f'truth_{args.type}.txt')

    eval_all(args.model_name,args.type,predict_path,truth_path,args.sbert_path,args.llama2_7b_chat_path,args.discriminator_ckpt_path,args.data_path,args.maxnewtoken,args.temperature,args.topp,args.batchsize)
    