from chacater_100_dataset import CharacterDataset
import random
import rich
random.seed(42)
from rich import print

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
    type_dict={
        0:'Zero-shot without limit',
        1:'Zero-shot with limit',
        2:'Few-shot without limit',
        3:'Few-shot with limit'}

    names, contexts, questions, answers = get_data('Data/test.txt')
    for type_index in range(4):
        print(f'[red]Type {type_dict[type_index]}[/red]')
        character_dataset=CharacterDataset(names, contexts, questions, answers, template_num=type_index, maxnewtoken=100)
        dataset_len=len(character_dataset)
        example_q, example_a = character_dataset[random.randint(0,dataset_len-1)]
        print('[green]Example question:')
        print(f'[cyan]{example_q}[/cyan]')
        print('[blue]Example answer:')
        print(f'[white]{example_a}')

