import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0,pc_valid=0.0, tasknum = 20):
    rootdata = "../data/"
    data={}
    taskcla=[]
    size=[3,32,32]

    if not os.path.isdir(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum)+"/"):
        os.makedirs(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum))
        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100(rootdata+'cifar/',train=True,download=True)
        dat['test']=datasets.CIFAR100(rootdata+'cifar/',train=False,download=True)
        for n in range(tasknum):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=100/tasknum
            data[n]['train']={'x': [],'y': [],'task_y': []}
            data[n]['test']={'x': [],'y': [],'task_y': []}
        
        for s in ['train','test']: 
            for x,y in dat[s]: 
                task_idx = int(y // (100/tasknum))
                data[task_idx][s]['x'].append(x)
                data[task_idx][s]['y'].append(y)
                data[task_idx][s]['task_y'].append(y)


        # "Unify" and save
        for t in range(tasknum):
            for s in ['train','test']:
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum)), 'data'+str(t+1)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum)), 'data'+str(t+1)+s+'y.bin'))
    
    # Load binary files
    data={}
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    #ids=list(shuffle(np.arange(tasknum),random_state=seed)+1)
    ids = list(np.arange(tasknum)+1)
    print('Task order =',ids)
    for i in range(tasknum):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum)), 'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(rootdata+'cifar_dataset/binary_split_cifar100_'+str(tasknum)),  'data'+str(ids[i])+s+'y.bin'))
            data[i][s]['task_y'] = data[i][s]['y']
        data[i]['ncla']=len(np.unique(data[i]['train']['y']))
        data[i]['name']='cifar100-'+str(ids[i-1])
            

    # Others
    n=0
    for t in range(tasknum):
        task_y = data[t]['train']['task_y']
        newtask_cls = list(np.unique(np.array(task_y)))
        sum_cls = max(data[t]['train']['y'])+1
        taskcla.append((t, newtask_cls, sum_cls))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
