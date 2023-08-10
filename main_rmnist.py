import argparse
import math 
import torch
from torchvision import transforms, datasets
from utils import TwoCropTransform,random_update,set_loader,set_weighted_loader,buffer_loader,MyDataset,Pad,Convert2RGB

from loss import SupConLoss
import copy
import sys, os, time
import numpy as np
import torch.nn as nn 
from torch import optim 
import pytorch_lightning as pl
import torch.nn.functional as F    
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
import wandb


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=2, help='seed')
    parser.add_argument('--wandb_ifuse', type=bool, default=False,
                        help='wandb_ifuse')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=100)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--optimizer', type=str, default="SGD", choices=['Adam', "SGD"],
                        help='optimizer for training')
    parser.add_argument('--lr_scheduler', type=str, default="Cos",
                        help='optimizer scheduler for training')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='rmnist',
                        choices=['rmnist','cifar10', 'cifar100', 'tiny-imagenet','mix_digit'], help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # Buffersize
    parser.add_argument('--Buffersize', type=int, default=500, help='Buffersize')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')
    parser.add_argument('--base_temperature', type=float, default=0.07, help='temperature 2 for loss function')

    # other setting
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    
    parser.add_argument('--weighted_loss', type=float, default=1,
                        help='weighted_loss')
    parser.add_argument('--data_use', type=str, default="all",
                        help='all vs memory')
    parser.add_argument('--map_pos', type=str, default="logit",
                        help='embed vs logit')
    parser.add_argument('--distill_loss', type=str, default="contrastive",
                        help='mse vs contrastive')
    parser.add_argument('--feat_dim', type=int, default=2048,
                        help='feat_dim')
    

    opt = parser.parse_args()

    opt.warmup_from = 0.01
    opt.warm_epochs = 10
    eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
            1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2

    if opt.dataset == 'cifar10':
        opt.total_n_cls = 10
        opt.tasknum = 5
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        opt.total_n_cls = 100
        opt.tasknum = 10
        opt.size = 32
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    elif opt.dataset == 'tiny-imagenet':
        opt.total_n_cls = 200
        opt.tasknum = 10
        opt.size = 64
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'mix_digit':
        opt.total_n_cls = 10
        opt.size = 32
        opt.tasknum = 7
        mean=[0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5]
    elif opt.dataset == 'rmnist':
        opt.total_n_cls = 10
        opt.size = 28
        opt.tasknum = 20
        mean=[0.5]
        std=[0.5]
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset == 'mix_digit':
        opt.train_transform =transforms.Compose([
                            transforms.RandomResizedCrop(size=opt.size, scale=(0.7, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            Pad(32),
                            Convert2RGB(3),
                            normalize,
                            ])
        opt.val_transform = transforms.Compose([
            transforms.ToTensor(),
            Pad(32),
            Convert2RGB(3),
            normalize
        ])
    elif opt.dataset == "rmnist":
        opt.train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(size=opt.size, scale=(0.7, 1.)),
                            transforms.ToTensor(),
                            ])

        opt.val_transform = transforms.Compose([
                            transforms.ToTensor(), 
                            ])
    else:
        opt.train_transform = transforms.Compose([
            transforms.Resize(size=(opt.size, opt.size)),
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([ transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                               p=0.5 if opt.size > 32 else 0.0), 
            transforms.ToTensor(),
            normalize,
        ])
        opt.val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    return opt


class Clf_model(pl.LightningModule):
    def __init__(self, opt, total_n_cls, sslmodel=None):
        super().__init__()
        self.opt = opt
        sslmodel.eval()
        self.sslmodel = sslmodel
        self.classifier = LinearClassifier(name=opt.model, num_classes=total_n_cls)
        self.precriterion = torch.nn.CrossEntropyLoss() 

    def configure_optimizers(self):
        if self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.classifier.parameters(),
                                        lr=1,
                                        #lr=self.opt.learning_rate,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(),
                                         lr=self.opt.learning_rate,
                                         weight_decay=self.opt.weight_decay)
        else:
            optimizer = optim.AdamW(self.classifier.parameters(),
                                    lr=self.opt.learning_rate)

        if self.opt.lr_scheduler == "Step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif self.opt.lr_scheduler == "Cos":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)
        else:
            return optimizer

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        feat, _ = self.sslmodel(x)
        output = self.classifier(feat.detach())
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]
        bsz = y.shape[0]
        output = self(x)
        output = output[:, :self.n_cls]
        preloss = self.precriterion(output[0:bsz], y)
        return preloss

    def test_step(self, batch, batch_idx):
        _, acc = self._get_preds_accuracy(batch)
        metrics = {"test_acc": acc}
        self.log('test_acc', acc)
        return metrics

    def _get_preds_accuracy(self, batch):
        x, y = batch
        output = self(x)
        output = output[:, :self.n_cls]
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y)
        preloss = self.precriterion(output, y)
        return preloss, acc

    def update_taskinfo(self, n_cls, sslmodel):
        self.n_cls = n_cls 
        self.sslmodel = sslmodel

class PL_model(pl.LightningModule):
    def __init__(self, opt, unique_cls=[]):
        super().__init__()
        self.opt = opt
        self.model = network(name=opt.model,feat_dim=opt.feat_dim)
        self.sslcriterion = SupConLoss(temperature=opt.temp,base_temperature=opt.base_temperature)
        self.distillcriterion = SupConLoss(temperature=opt.temp,base_temperature=opt.base_temperature,contrast_mode="all")
        self.unique_cls = unique_cls 

        if opt.map_pos == "embed":  
            self.distill_predictor =nn.Linear(512, 512)
        else:  
            self.distill_predictor =  nn.Linear(opt.feat_dim, opt.feat_dim)
        

    def configure_optimizers(self):
        learnable_parameters = [{'params': list(self.model.parameters())
                                +list(self.distill_predictor.parameters())}]
        if self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD(learnable_parameters,
                                        lr=self.opt.learning_rate,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == "Adam":
            optimizer = torch.optim.Adam(learnable_parameters,
                                         lr=self.opt.learning_rate,
                                         weight_decay=self.opt.weight_decay)
        else:
            optimizer = optim.AdamW(learnable_parameters,
                                    lr=self.opt.learning_rate)

        if self.opt.lr_scheduler == "Step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif self.opt.lr_scheduler == "Cos":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)
        else:
            return optimizer

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        feat, logits = self.model(x)
        return feat,logits


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.cat([x[0], x[1]], dim=0)
        bsz = y.shape[0]

        embed, z = self(x)
        if self.opt.map_pos == "logit":
            z1, z2 = torch.split(z, [bsz, bsz], dim=0)
        elif self.opt.map_pos == "embed": 
            z1, z2 = torch.split(embed, [bsz, bsz], dim=0)
        p1 = self.distill_predictor(z1)
        p2 = self.distill_predictor(z2)

        #logits = F.normalize(z, dim=1)
        logits = z
        f1, f2 = torch.split(logits, [bsz, bsz], dim=0)
        logits = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        sslloss = self.sslcriterion(logits, y, target_labels=self.newtask_cls) 
        self.log('train_ssl_loss', sslloss)

        dis_loss = 0
        if len(self.unique_cls_pre) > 0: 
            embed_old, z_old = self.old_model(x)
            if self.opt.distill_loss=="contrastive":
                #z_old = F.normalize(z_old, dim=1)
                #norm_p = F.normalize(torch.cat([p1, p2]), dim=1)
                norm_p = torch.cat([p1, p2])
                p1, p2 = torch.split(norm_p, [bsz, bsz], dim=0)
                
            if self.opt.map_pos == "logit":
                z1_old, z2_old = torch.split(z_old, [bsz, bsz], dim=0)
            elif self.opt.map_pos == "embed":
                z1_old, z2_old = torch.split(embed_old, [bsz, bsz], dim=0) 
 
            if self.opt.distill_loss=="mse":
                dis_loss += (F.mse_loss(p1, z1_old.clone()) +  F.mse_loss(p2, z2_old.clone())) / 2
            elif self.opt.distill_loss=="contrastive":
                sim1 = torch.cat([z1_old.unsqueeze(1),p1.unsqueeze(1)], dim=1) 
                dis_loss += 0.5*self.distillcriterion(sim1) 
                sim2 = torch.cat([z2_old.unsqueeze(1),p2.unsqueeze(1)], dim=1) 
                dis_loss += 0.5*self.distillcriterion(sim2) 
                 
                
        self.log('train_distill_loss', dis_loss)

        return sslloss + self.opt.weighted_loss*dis_loss


    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int): 
        ## warm up learning rate
        if self.current_epoch < self.opt.warm_epochs:
            total_batches = len(self.trainer.train_dataloader)
            p = (batch_idx + self.current_epoch * total_batches) / \
                (self.opt.warm_epochs * total_batches)
            lr = self.opt.warmup_from + p * (self.opt.warmup_to - self.opt.warmup_from)
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = lr
 
    def update_taskinfo(self, newtask_cls):
        self.unique_cls_pre = self.unique_cls 
        self.unique_cls = list(np.unique(self.unique_cls_pre+newtask_cls))
        self.newtask_cls = newtask_cls
        #print("===== updated n_cls: ", self.unique_cls) 
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()


def main():
    opt = parse_option()
    pl.seed_everything(opt.seed)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    wandb_ifuse =False
    if wandb_ifuse:
        wandb.init(project=opt.dataset, entity="nan1")
        wandb_logger = WandbLogger(project=opt.dataset)

    if opt.dataset == 'cifar10':
        from dataloaders import split_cifar10 as dataloader 
    elif opt.dataset == 'tiny-imagenet':
        from dataloaders import tinyimg as dataloader
    elif opt.dataset == 'mix_digit':
        from dataloaders import mixdigit as dataloader
    elif opt.dataset == 'rmnist':
        from dataloaders import rmnist as dataloader

    if opt.dataset == 'rmnist':
        from networks.mlp import SupConMLP as network
        from networks.mlp import LinearClassifier
    else:
        from networks.resnet import SupConResNet_pl_non_nom as network
        from networks.resnet import LinearClassifier
    global network, LinearClassifier
        


    data, taskcla, _ = dataloader.get(seed=opt.seed, tasknum=opt.tasknum)
    traindata_all = []
    for t, _,_ in taskcla:
        taskdata = []
        for i in range(len(data[t]['train']['x'])):
            taskdata.append([data[t]['train']['x'][i], data[t]['train']['y'][i],  data[t]['train']['task_y'][i]])
        traindata_all.append(taskdata)
 
    Buffer = []
    model = PL_model(opt, unique_cls=[])
    model_clf = Clf_model(opt, total_n_cls=opt.total_n_cls, sslmodel=model)
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath="./callback_ckpt", every_n_epochs=1),
                pl.callbacks.LearningRateMonitor(logging_interval='step')]

    if wandb_ifuse:
        wandb_logger.watch(model)

    # Loop tasks
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    for t, newtask_cls, sum_cls in taskcla: 
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']),newtask_cls,sum_cls)
        print('*' * 100)

        train_loader, _ = set_loader(opt, data[t], Buffer)
        weight_train_loader = set_weighted_loader(opt, data[t], Buffer)

        model.update_taskinfo(newtask_cls)

        # build trainer  
        if n_gpu > 1:
            if wandb_ifuse:
                if t>0:
                    trainer = pl.Trainer(gpus=-1,  logger=wandb_logger, max_epochs=opt.epochs,strategy='ddp',log_every_n_steps=1,
                        #accumulate_grad_batches=4,
                        replace_sampler_ddp=False, precision=16)
                        #, callbacks=callbacks)
                else:
                    trainer = pl.Trainer(gpus=-1, max_epochs=opt.start_epoch,strategy='ddp',log_every_n_steps=1,replace_sampler_ddp=False,logger=wandb_logger, 
                        #accumulate_grad_batches=4,
                     precision=16)
                     #, callbacks=callbacks) 
            else:
                if t > 0:
                    trainer = pl.Trainer(gpus=-1, max_epochs=opt.epochs,strategy='ddp',log_every_n_steps=1,
                        replace_sampler_ddp=False, precision=16, callbacks=callbacks)
                else:
                    trainer = pl.Trainer(gpus=-1, max_epochs=opt.start_epoch,strategy='ddp',log_every_n_steps=1,replace_sampler_ddp=False,
                                     precision=16, callbacks=callbacks)
        else:
            if wandb_ifuse:
                if t > 0:
                    trainer = pl.Trainer(gpus=-1, logger=wandb_logger, max_epochs=opt.epochs, 
                        #accumulate_grad_batches=4,
                                         progress_bar_refresh_rate=1,
                                         precision=16, callbacks=callbacks, log_every_n_steps=1)
                else:
                    trainer = pl.Trainer(gpus=-1, logger=wandb_logger, max_epochs=opt.start_epoch,
                    # accumulate_grad_batches=4,
                                         progress_bar_refresh_rate=1,
                                         precision=16, callbacks=callbacks, log_every_n_steps=1)
            else:
                trainer = pl.Trainer(gpus=-1, max_epochs=opt.epochs, progress_bar_refresh_rate=1, precision=16,
                                    callbacks=callbacks)
                #trainer = pl.Trainer(max_epochs=opt.epochs, callbacks=callbacks)

        trainer_clf = pl.Trainer(gpus=-1, max_epochs=opt.start_epoch, precision=16,strategy='ddp',log_every_n_steps=1,replace_sampler_ddp=False, 
            #accumulate_grad_batches=4
            )
 
        # Train
        trainer.fit(model, train_loader)
        if t>len(taskcla)-2: 
            model_clf.update_taskinfo(sum_cls, sslmodel=model)
            trainer_clf.fit(model_clf, weight_train_loader)

            # Test
            for u in range(t + 1):
                _, test_loader = set_loader(opt, data[u])
                test_metric = trainer_clf.test(dataloaders=test_loader)
                print("task", u, test_metric)
                acci = test_metric[0].get("test_acc")
                acc[t, u] = acci

        print("Random memory update")
        Buffer = random_update(t, traindata_all, Buffer=Buffer, buffersize=opt.Buffersize)

    print(acc)
    print("ACC mean:", np.mean(acc[opt.tasknum-1]))
    if wandb_ifuse:
        wandb.log({"Acc_Mean":np.mean(acc[opt.tasknum-1])})
        wandb.finish()




if __name__ == '__main__':
    main()

