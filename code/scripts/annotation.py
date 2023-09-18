from re import L
import sys
import os
import pandas as pd

def mapping_table():
    mp={}
    with open('data/ntu_dark/mapping_table.txt','r') as f:
        content = f.readlines()
    for item in content:
        elem = item[:-1].split('\t')
        mp[elem[1]]=elem[0]
    return mp

def train_file_list():
    mp=mapping_table()
    source='data/ntu_dark/rawframes_train'
    action_dirs = os.listdir(source)
    with open('data/ntu_dark/ntu_dark_train_list.txt','w') as f:
        for ac in action_dirs:
            files=  os.listdir(os.path.join(source,ac))
            # print(files)
            for file in files:
                # print(os.path.join(source,ac,file),end='\t\t')
                # print(mp[ac])
                f.writelines(os.path.join(source,ac,file)+' '+mp[ac]+'\n')
    pass

def valid_file_list():
    content = pd.read_csv('data/ntu_dark/ntu_dark_val_list.csv')
    with open('data/ntu_dark/ntu_dark_test_list.txt','w') as f:
        for i in range(len(content)):
            video_path = os.path.join('data/ntu_dark/rawframes_val',content['Video'][i])
            f.writelines(video_path+' '+str(content['ClassID'][i])+'\n')
    pass

if __name__=='__main__':
    valid_file_list()
    pass