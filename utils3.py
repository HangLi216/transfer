import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset
import math
import random
import torch.nn.functional as F 
from PIL import Image


class Pad(object):
    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # If the H and W of img is not equal to desired size,
        # then pad the channel of img to desired size.
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)


class Convert2RGB(object):
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, img):
        # If the channel of img is not equal to desired size,
        # then expand the channel of img to desired size.
        img_channel = img.size()[0]
        img = torch.cat([img] * (self.num_channel - img_channel + 1), 0)
        return img



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if isinstance(x, str):
            with open(x, "rb") as f:
                x = Image.open(f)
                x.convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def random_update(t, traindata_all, Buffer, buffersize=500):  
    new_task_y = [traindata_all[t][k][2] for k in range(len(traindata_all[t]))]
    pre_task_y = [Buffer[k][2] for k in range(len(Buffer))] 
    unique_cls = np.unique(new_task_y+pre_task_y)
    if t > 0:
        pre_cls = max(np.unique(pre_task_y))+1
    else:
        pre_cls=0

    new_selectid = []
    buffer_selectid=[]

    for c_idx, c in enumerate(unique_cls):
        size_for_c_float = ((buffersize - len(buffer_selectid) - len(new_selectid)) / (len(unique_cls) - c_idx))
        p = size_for_c_float -  ((buffersize - len(buffer_selectid) - len(new_selectid)) // (len(unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        
        if c < pre_cls:
            mask = pre_task_y == c
            while True:
                try:
                    s = random.sample(np.argwhere(mask==True).tolist(), size_for_c)
                    break
                except:
                    size_for_c -=1
                
            buffer_selectid += s 
        else:
            mask = new_task_y == c
            try:
                s = random.sample(np.argwhere(mask==True).tolist(), size_for_c)
            except:
                s = np.argwhere(mask==True).tolist()
            new_selectid += s
            print(c, len(s)) 
 
    if t > 0:
        Buffer = [Buffer[k[0]] for k in buffer_selectid]
    buffxy = [traindata_all[t][k[0]] for k in new_selectid]
    Buffer += buffxy
    return Buffer  


def score_update(t, traindata_all, Buffer, opt, sslmodel=None):
    device = (torch.device('cuda')
                if next(sslmodel.parameters()).is_cuda
                else torch.device('cpu'))

    tem_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

    buffersize=opt.Buffersize
    number = int(buffersize / (t + 1))
    remain_number = buffersize - number
    if t > 0:
        selectid = random.sample(range(len(Buffer)), remain_number)
        Buffer = [Buffer[k] for k in selectid] 
    x = [traindata_all[t][k][0] for k in range(len(traindata_all[t]))]
    y = [traindata_all[t][k][1] for k in range(len(traindata_all[t]))] 
    Dataset = MyDataset(x, y, transform=TwoCropTransform(tem_transform))
    Dataloader = torch.utils.data.DataLoader(
        Dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    all_sim = []
    for batch_x, batch_y in Dataloader:
        bsz = batch_y.shape[0]
        batch_2x = torch.cat([batch_x[0], batch_x[1]], dim=0).to(device)
        _ ,logits =  sslmodel(batch_2x)
        f1, f2 = torch.split(logits, [bsz, bsz], dim=0) 
        feature_sim = torch.matmul(f1, f2.T)
        feature_sim = torch.diagonal(feature_sim, 0)
        all_sim = all_sim + list(feature_sim.tolist())
    sorted_id = np.argsort(all_sim)
    selectid = sorted_id[0:number]
    buffxy = [traindata_all[t][k] for k in selectid]
    Buffer += buffxy
    return Buffer



def set_loader(opt, data,Buffer=[]):
    # construct data loader
    x = data['train']['x']
    task_y = data['train']['task_y']
    if len(Buffer)>0:
        selectid = range(len(Buffer))
        samplex = [Buffer[k][0] for k in selectid]
        sample_task_y = [Buffer[k][2] for k in selectid]
        x = x+samplex
        task_y = task_y+sample_task_y
    train_dataset = MyDataset(x, task_y, transform=TwoCropTransform(opt.train_transform))
    test_dataset = MyDataset(data['test']['x'], data['test']['y'], transform=opt.val_transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, test_loader


def buffer_loader(opt, Buffer): 
    selectid = range(len(Buffer))
    samplex = [Buffer[k][0] for k in selectid]
    sampley = [Buffer[k][1] for k in selectid]
    Buffer_dataset = MyDataset(samplex, sampley, transform=TwoCropTransform(opt.train_transform))

    loader = torch.utils.data.DataLoader(Buffer_dataset, batch_size=opt.batch_size, shuffle=True,
                                         num_workers=opt.num_workers, pin_memory=True)

    return loader


def set_weighted_loader(opt, data, Buffer=[]):
    x = data['train']['x']
    y = data['train']['y']
    task_y = data['train']['task_y']

    if len(Buffer)>0:
        selectid = range(len(Buffer))
        samplex = [Buffer[k][0] for k in selectid]
        sampley = [Buffer[k][1] for k in selectid]
        sample_task_y = [Buffer[k][1] for k in selectid]

        x = x+samplex
        y = y+sampley
        task_y = task_y+ sample_task_y

    train_dataset = MyDataset(x, y, transform=TwoCropTransform(opt.train_transform))

        
    ut, uc = np.unique(np.array(task_y), return_counts=True)
    weights = np.array([0.] * len(task_y))
    for t, c in zip(ut, uc):
        weights[task_y == t] = 1. / c

    from torch.utils.data import WeightedRandomSampler
    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True,sampler=train_sampler)

    return train_loader
