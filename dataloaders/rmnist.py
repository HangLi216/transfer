import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

def get(seed=12, tasknum = 10):
    dataloadfile = "savedata/rmnist_tasknum_{}_seed{}.npz".format(tasknum,seed)
    data = {}
    taskcla = []
    size = [1, 28, 28]
    datadir = "../data/mnist_usps_svhn/MNIST/MNIST/"
    dat = {}
    dat['train'] = datasets.MNIST(datadir, train=True, download=True)
    dat['test'] = datasets.MNIST(datadir, train=False, download=True)

    if os.path.exists(dataloadfile):
        rotload = np.load(dataloadfile)
        allrot = rotload["allrot"]
    else:
        allrot = []
    print(allrot)

    for i in range(tasknum):
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'rotate_mnist-{:d}'.format(i)
        data[i]['ncla'] = 10

        if os.path.exists(dataloadfile):
            rot = allrot[i]
        else:
            min_rot = 1.0 * i / tasknum * 180
            max_rot = 1.0 * (i + 1) / tasknum * 180
            # rot = random.random() * (max_rot - min_rot) + min_rot
            rot = np.random.uniform(0, 180)
            allrot.append(rot)
        print(i,rot, end=',') 


        for s in ['train', 'test']:
            if s == 'train':
                arr = rotate_dataset(dat[s].train_data, rot)
                label = dat[s].train_labels.tolist()
            else:
                arr = rotate_dataset(dat[s].test_data,rot)
                label = dat[s].test_labels.tolist()

            data[i][s]={}
            data[i][s]['x'] = arr
            data[i][s]['y'] = label
            data[i][s]['task_y'] = list(np.asarray(label) + i*10)


    # Others
    n=0
    for t in range(tasknum):
        task_y = data[t]['train']['task_y']
        newtask_cls = list(np.unique(np.array(task_y)))
        sum_cls = max(data[t]['train']['y'])+1
        taskcla.append((t, newtask_cls, sum_cls))
        n+=data[t]['ncla']
    data['ncla']=n

    if not os.path.exists(dataloadfile):
        np.savez(dataloadfile, allrot=allrot)
    return data, taskcla, size


def rotate_dataset(d, rotation):
    result = [] 
    for i in range(d.size(0)):
        img = Image.fromarray(d[i].numpy(), mode='L')
        result.append(img.rotate(rotation))
        #result[i] = transform(img.rotate(rotation)).view(1,28,28)
        #result[i] = tensor(img.rotate(rotation)).view(784)
    return result
########################################################################################################################
