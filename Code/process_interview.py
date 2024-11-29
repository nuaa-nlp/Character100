import os
from collections import defaultdict

def get_name_job_dict(root_path):
    name2job=defaultdict(str)
    job2name=defaultdict(list)
    people_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'people')))]
    print(f'People count:{len(people_names)}')
    actor_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'actors')))]
    athletes_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'athletes')))]
    criminal_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'criminals')))]
    political_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'political')))]
    singers_names=[item.replace('.txt','') for item in sorted(os.listdir(os.path.join(root_path,'singers')))]
    for name in people_names:
        if name in actor_names:
            name2job[name]='actor'
            job2name['actor'].append(name)
        elif name in athletes_names:
            name2job[name]='athletes'
            job2name['athletes'].append(name)
        elif name in criminal_names:
            name2job[name]='criminal'
            job2name['criminal'].append(name)
        elif name in political_names:
            name2job[name]='political'
            job2name['political'].append(name)
        elif name in singers_names:
            name2job[name]='singers'
            job2name['singers'].append(name)
        else:
            name2job[name]='N/A'
            job2name['N/A'].append(name)
    for k,v in name2job.items():
        print(k,v)
    for k,v in job2name.items():
        print(k,v)
    
    return name2job, job2name, people_names


if __name__ == '__main__':
    name2job, job2name, people_names=get_name_job_dict('Data/raw_data')

    origin_path='/data/wangx/roleplay/Character100_git/Data/interviews_origin'
    processed_path='/data/wangx/roleplay/Character100_git/Data/interviews_processed'

    origin_filelists=os.listdir(origin_path)
    processed_filelists=os.listdir(processed_path)

    origin_filelists.sort()
    processed_filelists.sort()
    assert len(origin_filelists)==len(processed_filelists)
    
    for origin_file, processed_file in zip(origin_filelists, processed_filelists):
        with open(os.path.join(origin_path, origin_file), 'r') as f1:
            origin_data=f1.readlines()
        with open(os.path.join(processed_path, processed_file), 'r') as f2:
            processed_data=f2.readlines()
        missed=set(origin_data)-set(processed_data)
        assert origin_file==processed_file
        print(origin_file.replace('.txt',''),name2job[origin_file.replace('.txt','')],len(set(origin_data)), len(set(processed_data)),len(missed))
        if len(name2job[origin_file.replace('.txt','')])==0:
            print('^@@')

        with open(os.path.join('/data/wangx/roleplay/Character100_git/Data/interviews_unused', origin_file), 'w') as f3:
            for item in missed:
                f3.write(item.strip()+'\n')