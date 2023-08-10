import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import torch.nn.functional as F
import json
########################################################################################################################

def get(seed=0, tasknum = 10):
    path = "../data/TINYIMG/"
    data = {}
    tasknum = 10 
    if not os.path.isdir(path+'True'):
        traindatax = []
        for num in range(20):
            traindatax.append(np.load(os.path.join(
                path, 'processed/x_%s_%02d.npy' %  ('train', num + 1))))
        traindatax = np.concatenate(np.array(traindatax))

        traindatay = []
        for num in range(20):
            traindatay.append(np.load(os.path.join(
                path, 'processed/y_%s_%02d.npy' % ('train', num + 1))))
        traindatay = np.concatenate(np.array(traindatay))

        testdatax = []
        for num in range(20):
            testdatax.append(np.load(os.path.join(
                path, 'processed/x_%s_%02d.npy' % ('val', num + 1))))
        testdatax = np.concatenate(np.array(testdatax))

        testdatay = []
        for num in range(20):
            testdatay.append(np.load(os.path.join(
                path, 'processed/y_%s_%02d.npy' % ('val', num + 1))))
        testdatay = np.concatenate(np.array(testdatay))

        for i in range(tasknum):
            data[i] = {}
            data[i]['name'] = 'tinyimg-{:d}'.format(i)
            data[i]['ncla'] = 20
            data[i]['train']={'x': [],'y': [],'task_y': []}
            data[i]['test']={'x': [],'y': [],'task_y': []}

        for s in ['train', 'test']:
            if s == 'train':
                for idx in range(len(traindatay)):
                    task_idx = traindatay[idx] // 20 

                    img = Image.fromarray(np.uint8(255 * traindatax[idx]))
                    data[task_idx][s]['x'].append(img)
                    data[task_idx][s]['y'].append(traindatay[idx])
                    data[task_idx][s]['task_y'].append(traindatay[idx])

            if s == 'test':
                for idx in range(len(testdatay)):
                    task_idx = testdatay[idx] // 20
                    # data[task_idx][s]['x'].append(testdatax[idx])
                    # data[task_idx][s]['y'].append(testdatay[idx]%10)
                    # #
                    img = Image.fromarray(np.uint8(255 * testdatax[idx]))
                    data[task_idx][s]['x'].append(img)
                    data[task_idx][s]['y'].append(testdatay[idx])


    # Others
    taskcla = []
    n = 0
    for t in range(tasknum):
        task_y = data[t]['train']['task_y']
        newtask_cls = list(np.unique(np.array(task_y)))
        sum_cls = max(data[t]['train']['y'])+1
        taskcla.append((t, newtask_cls, sum_cls))
        # taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, None


########################################################################################################################
