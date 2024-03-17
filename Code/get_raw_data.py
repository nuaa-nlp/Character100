import requests
import os
import re
from lxml import etree, html
from tqdm import tqdm

def remove_square_brackets(text):
    pattern = r'\[\d+\]'
    result = re.sub(pattern, '', text)
    return result

def getContent(name, category, link):
    response = requests.get(url=link,)
    tree = html.fromstring(response.text)
    ps = tree.xpath('/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/p')
    if not os.path.exists(f'./data_path/{category}'):
        os.mkdir(f'./data_path/{category}')
    with open(f'./data_path/{category}/{name}.txt','w') as f:
        for p in ps:
            text = remove_square_brackets(p.text_content()).strip()
            if text != '':
                f.write(text+'\n')

def getByXpath(myXpath, category):
    print(f'Processing {category}...')
    response = requests.get(url="https://en.wikipedia.org/wiki/Wikipedia:Popular_pages",)
    tree = html.fromstring(response.text)
    res = tree.xpath(myXpath)
    with open(f'./data_path/{category}_names.txt','w') as f:
        for item in tqdm(res[1:]):
            link = item.xpath('*/a/@href')
            name = item.xpath('*/a/@title')[0].replace(' ', '_')
            link = 'https://en.wikipedia.org' + link[0]
            getContent(name,category,link)
            f.write(name+'\n')

if __name__ == '__main__':
    peopleXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[10]/tbody/tr'
    sigeresXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[11]/tbody/tr'
    actorsXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[12]/tbody/tr'
    athletesXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[14]/tbody/tr'
    politicalXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[15]/tbody/tr'
    criminalsXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[16]/tbody/tr'
    premodernXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[17]/tbody/tr'
    thirdMillenniumXpath = '/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/table[18]/tbody/tr'
    if not os.path.exists('./data_path'):
        os.mkdir('./data_path')
    getByXpath(peopleXpath, 'people')
    # getByXpath(sigeresXpath, 'singers')
    # getByXpath(actorsXpath, 'actors')
    # getByXpath(athletesXpath, 'athletes')
    # getByXpath(politicalXpath, 'political')
    # getByXpath(criminalsXpath, 'criminals')
    # getByXpath(premodernXpath, 'premodern')
    # getByXpath(thirdMillenniumXpath, 'thirdMillennium')
    
    
