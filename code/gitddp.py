import csv
import numpy as np
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# train_accuracy_values = np.array([])
# validation_accuracy_values = np.array([])

N_CLASSES = 10

def create_analog_network():
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 800, 128]
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=5, stride=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(in_features=channel[2], out_features=channel[3]),
        nn.Tanh(),
        nn.Linear(in_features=channel[3], out_features=N_CLASSES),
        nn.LogSoftmax(dim=1)
    )
    return model

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--data', metavar='DIR', default='./data', help='path to dataset')
# TODO modify the default path here!

parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.005,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs # this average is necessary!
    return rt


def main():

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8004'
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    dist.init_process_group(backend='nccl', world_size=args.nprocs, rank=local_rank)
    
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    model = create_analog_network()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = MultiStepLR(optimizer, milestones=[50, 90, 105], gamma=0.2)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.491, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616])

    train_transforms = transforms.Compose([transforms.RandomCrop((32, 32), padding=4), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR10(traindir, train=True, download=True, transform=train_transforms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    val_dataset = datasets.CIFAR10(valdir, train=False, download=True, transform=test_transforms)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,num_workers=2,pin_memory=True,sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    train_accuracy_values = np.array([])
    validation_accuracy_values = np.array([])
    total_time_array = np.array([]) 
    total_time = 0

    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, args)
        start_train_time = time.time()
        train_acc = train(train_loader, model, criterion, optimizer, epoch, local_rank, args)
        end_train_time = time.time()

        val_acc1 = validate(val_loader, model, criterion, local_rank, args)

        scheduler.step()

        train_accuracy_values = np.append(train_accuracy_values, train_acc.cpu().numpy())
        validation_accuracy_values = np.append(validation_accuracy_values, val_acc1)

        total_time += (end_train_time - start_train_time)

        total_time_array = np.append(total_time_array, total_time)
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

    # train_accuracy_values.cpu()
    # validation_accuracy_values.cpu()

    np.save("train_accuracy_2torch.npy", train_accuracy_values)
    np.save("validation_accuracy_2torch.npy",validation_accuracy_values)
    np.save("total_time_per_epoch_2torch.npy", total_time_array/60)

    print("Total Time  = {:.3}\n".format(total_time/60)) 


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, local_rank, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return acc1
    # train_accuracy_values.append(acc1.numpy())



def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, local_rank, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
    
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('* Acc@1 {top1.avg:.3f}'.format(top1=top1))

        # validation_accuracy_values = np.append(validation_accuracy_values, top1.numpy())
        # validation_accuracy_values.append(top1.numpy())

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = state['arch'] + '.' + filename
    torch.save(state, filename)
    if is_best:
        filename2 = state['arch'] + '.model_best.pth.tar'
        shutil.copyfile(filename, filename2)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, local_rank, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        #maxk = max(topk)
        #batch_size = target.size(0)

        #_, pred = output.topk(maxk, 1, True, True)
        #pred = pred.t()
        #correct = pred.eq(target.view(1, -1).expand_as(pred))

        #res = []
        #for k in topk:
        #    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #    res.append(correct_k.mul_(100.0 / batch_size))
        predy = torch.max(output, 1)[1].data.squeeze()
        acc = (predy == target).sum().item()/float(target.size(0))
        acc = torch.tensor(acc).cuda(local_rank)
        res = []
        res.append(acc)
        res.append(acc)
        return res

def accuracy2(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

