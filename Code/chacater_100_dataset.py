import torch
from torch.utils.data import Dataset
import os

def get_prompt(name, context, question, template_num = 0, maxnewtoken = 100):
    prompts=[f"""
    Imagine you are {name}, you need to role-play as he/she, and your basic information is as follows: {context} Now you need to answer the question "{question}", and as the person you need to role-play, your answer is:
    """,
    f"""
    Imagine you are {name}, you need to role-play as he/she, and your basic information is as follows: {context} Now you need to answer the question "{question}", and as the person you need to role-play, your answer (no more than {maxnewtoken} words) to this question should be as follows: 
    """,
    f"""
    Imagine you are {name}, you need to role-play as he/she, and your basic information is as follows: {context} 
    Example: Imaging you are Neymar, the basic information is Neymar continued his ascendancy in 2010, and, on 15 April 2010, he scored five goals for Santos in an 8–1 rout of Guarani in the qualifying stages of the Brazilian Cup. Following the 2010 Campeonato Paulista in which Neymar scored 14 goals in 19 games, the club were crowned champions after a 5–5 aggregate win over Santo André in the finals. Neymar was subsequently given the award for the best player in the competition. The question is "In which year did Neymar score five goals for Santos in an 8–1 victory over Guarani during the Brazilian Cup qualifying stages?" The answer to this question is Neymar scored five goals for Santos in an 8–1 victory over Guarani during the Brazilian Cup qualifying stages on 15 April 2010.
    Now you need to answer the question "{question}", and as the person you need to role-play, your answer is:
    """,
    f"""
    Imagine you are {name}, you need to role-play as he/she, and your basic information is as follows: {context} 
    Example: Imaging you are Neymar, the basic information is Neymar continued his ascendancy in 2010, and, on 15 April 2010, he scored five goals for Santos in an 8–1 rout of Guarani in the qualifying stages of the Brazilian Cup. Following the 2010 Campeonato Paulista in which Neymar scored 14 goals in 19 games, the club were crowned champions after a 5–5 aggregate win over Santo André in the finals. Neymar was subsequently given the award for the best player in the competition. The question is "In which year did Neymar score five goals for Santos in an 8–1 victory over Guarani during the Brazilian Cup qualifying stages?" The answer to this question is Neymar scored five goals for Santos in an 8–1 victory over Guarani during the Brazilian Cup qualifying stages on 15 April 2010.
    Now you need to answer the question "{question}", and as the person you need to role-play, your answer (no more than {maxnewtoken} words) to this question should be as follows: 
    """
    ]
    return prompts[template_num].strip()

class CharacterDataset(Dataset):
    def __init__(self, names, contexts, questions, answers, template_num = 0, maxnewtoken = 100) -> None:
        super().__init__()
        self.names = names
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.template_num = template_num
        self.maxnewtoken = maxnewtoken
    
    def __getitem__(self, index):
        return get_prompt(self.names[index], self.contexts[index], self.questions[index], self.template_num, self.maxnewtoken), self.answers[index]
    
    def __len__(self):
        return len(self.names)

class DiscriminatorDataset(Dataset):
    def __init__(self, names, contexts, questions, answers, maxnewtoken = 100) -> None:
        super().__init__()
        self.names = names
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.maxnewtoken = maxnewtoken
    
    def __getitem__(self, index):
        return self.answers[index], self.names[index]
    
    def __len__(self):
        return len(self.names)

class DiscriminatorEvalDataset(Dataset):
    def __init__(self, predicts) -> None:
        super().__init__()
        self.predicts = predicts

    def __getitem__(self, index):
        return self.predicts[index]
    
    def __len__(self):
        return len(self.predicts)