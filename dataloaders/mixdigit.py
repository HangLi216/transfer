import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import torch.nn.functional as F


def get(seed=0, tasknum = 7):
    
    path = '../data/mnist_usps_svhn/'
    tasknum = 7

    MNIST_train = datasets.MNIST(root=os.path.join(path, "MNIST"), 
                                        train=True,
                                        download=True)
    MNIST_test = datasets.MNIST(root=os.path.join(path, "MNIST"), 
                                 train=False,
                                 download=True)

    SVHN_train = datasets.SVHN(root=os.path.join(path, "SVHN"), 
                                      split='train',
                                      download=True)
    SVHN_test = datasets.SVHN(root=os.path.join(path, "SVHN"), 
                               split='test',
                               download=True)

    USPS_train = datasets.USPS(root=os.path.join(path, "USPS"), 
                                      train=True,
                                      download=True)
    USPS_test = datasets.USPS(root=os.path.join(path, "USPS"), 
                               train=False,
                               download=True)

    data = {}
    for i in range(tasknum):
        data[i] = {}
        data[i]['name'] = 'Digit_Task-{:d}'.format(i)
#         data[i]['ncla'] = 0
        data[i]['train'] = {'x': [], 'y': [], 'task_y': []}
        data[i]['test'] = {'x': [], 'y': []}

    ### train data
    MNIST_train_data = MNIST_train.train_data
    MNIST_train_label= MNIST_train.train_labels
    MNIST_train_idx = MNIST_train_label//2

    SVHN_train_data = SVHN_train.data 
    SVHN_train_label = SVHN_train.labels
    SVHN_train_idx = 1+SVHN_train_label// 2

    USPS_train_data = USPS_train.data
    USPS_train_label = USPS_train.targets
    USPS_train_idx = 2+(np.array(USPS_train_label)//2) 
    
    for idx, task_idx in enumerate(MNIST_train_idx.tolist()): 
        data[task_idx]['train']['x'].append(Image.fromarray(MNIST_train_data[idx].numpy(), mode='L'))
        data[task_idx]['train']['y'].append(MNIST_train_label[idx].item())
        data[task_idx]['train']['task_y'].append(MNIST_train_label[idx].item()+ task_idx*10)

    
    for idx, task_idx in enumerate(SVHN_train_idx.tolist()): 
        data[task_idx]['train']['x'].append(Image.fromarray(np.transpose(SVHN_train_data[idx], (1, 2, 0))))
        data[task_idx]['train']['y'].append(SVHN_train_label[idx].item())
        data[task_idx]['train']['task_y'].append(SVHN_train_label[idx].item()+ task_idx*10)


    for idx, task_idx in enumerate(USPS_train_idx.tolist()): 
        data[task_idx]['train']['x'].append(Image.fromarray(USPS_train_data[idx], mode='L'))
        data[task_idx]['train']['y'].append(USPS_train_label[idx])
        data[task_idx]['train']['task_y'].append(USPS_train_label[idx]+ task_idx*10)

    ### test data
    MNIST_test_data = MNIST_test.data
    MNIST_test_label= MNIST_test.targets
    MNIST_test_idx = MNIST_test_label//2

    SVHN_test_data = SVHN_test.data 
    SVHN_test_label = SVHN_test.labels
    SVHN_test_idx = 1+SVHN_test_label// 2

    USPS_test_data = USPS_test.data
    USPS_test_label = USPS_test.targets
    USPS_test_idx = 2+(np.array(USPS_test_label)//2)

    for idx, task_idx in enumerate(MNIST_test_idx.tolist()): 
        data[task_idx]['test']['x'].append(Image.fromarray(MNIST_test_data[idx].numpy(), mode='L'))
        data[task_idx]['test']['y'].append(MNIST_test_label[idx].item())
    
    for idx, task_idx in enumerate(SVHN_test_idx.tolist()): 
        data[task_idx]['test']['x'].append(Image.fromarray(np.transpose(SVHN_test_data[idx], (1, 2, 0))))
        data[task_idx]['test']['y'].append(SVHN_test_label[idx].item())

    for idx, task_idx in enumerate(USPS_test_idx.tolist()): 
        data[task_idx]['test']['x'].append(Image.fromarray(USPS_test_data[idx], mode='L'))
        data[task_idx]['test']['y'].append(USPS_test_label[idx]) 


#     task_y = []
#     for t in range(tasknum): 
#         task_y += data[t]['train']['task_y']
#     ut, uc = np.unique(np.array(task_y), return_counts=True) 


    # Others 
    taskcla = []
    for t in range(tasknum):
        task_y = data[t]['train']['task_y']
        newtask_cls = list(np.unique(np.array(task_y)))
        sum_cls = max(data[t]['train']['y'])+1
        taskcla.append((t, newtask_cls, sum_cls)) # task, current task new unique_class, max category
# taskcla.append((t, uc, max(data[t]['train']['y'])))
    return data, taskcla, None
