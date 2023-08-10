import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def get(seed=0, tasknum = 10):
    path = '../data/OfficeHome/'
    tasknum = 10
    data = {}
    for i in range(tasknum):
        data[i] = {}
        data[i]['name'] = 'Office_Task-{:d}'.format(i)
        data[i]['train'] = {'x': [], 'y': [], 'task_y': []}
        data[i]['test'] = {'x': [], 'y': []}

    domain = ["Art","Clipart","Product","RealWorld"] 
    

    ### train data
    
    for pos, d in enumerate(domain):
        folder = path + d 
        onedata = datasets.ImageFolder(root=folder)
        # Specify the indices of the images to be used for train and test set
        train_indices = list(range(0, int(len(onedata) * 0.8)))
        test_indices = list(range(int(len(onedata) * 0.8), len(onedata)))
        
        # Create the train and test sets
        train_set = Subset(onedata, train_indices)
        test_set = Subset(onedata, test_indices) 

#         train_idx = list(map(lambda x: x // 10+pos, train_set.dataset.targets))
#         print(np.max(train_idx),np.min(train_idx))

        for idx, task_idx in enumerate(train_set.dataset.targets):
            task_idx = task_idx // 10 + pos
            data[task_idx]['train']['x'].append(train_set.dataset.imgs[idx][0])
            data[task_idx]['train']['y'].append(train_set.dataset.targets[idx])
            data[task_idx]['train']['task_y'].append(train_set.dataset.targets[idx]+ task_idx*65)

        for idx, task_idx in enumerate(test_set.dataset.targets):
            task_idx = task_idx // 10 + pos
            data[task_idx]['test']['x'].append(test_set.dataset.imgs[idx][0])
            data[task_idx]['test']['y'].append(test_set.dataset.targets[idx])

    # Others 
    taskcla = []
    for t in range(tasknum):
        task_y = data[t]['train']['task_y']
        newtask_cls = list(np.unique(np.array(task_y)))
        print(np.min(task_y),np.max(task_y),newtask_cls)
        sum_cls = max(data[t]['train']['y'])+1
        taskcla.append((t, newtask_cls, sum_cls)) # task, current task new unique_class, max category 
    return data, taskcla, None

