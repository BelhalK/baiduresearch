import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import Imagefolder_train_val
import sys
import tensorboardX
from resnet import *
from logger import Logger, savefig
from termcolor import colored

# from opt.QAdam import QAdam
# from opt.compams import CompAMS
from opt.fedlamb import FedLAMB
from opt.Signum_SGD import SGD_distribute

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR', help='path to dataset')
# data can be ~/data/ILSVRC/Data/CLS-LOC or ~/data/tiny-imagenet-200/

parser.add_argument("--optimizer",type=str,default="signum",help="optimizer to use (sgd, damsgrad)",
    choices=["fedlamb", "fedams", "fedsgd", "fedlars"])   
parser.add_argument("--dataset",type=str,default="mnist",help="dataset to use",
    choices=["mnist", "imagenet", "tinyimagenet", "cifar"])   
parser.add_argument('--test_evaluate', action='store_true', help='Initiate test evaluation')
# parser.add_argument('--method', default='full', type=str, 
#     choices=['topk','unif','tong','sign','full','QAdam','SGD','1bitAdam'], help='compression algorithm')


parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--beta2', default=0.999, type=float, help='betar2 EMA var scale for ADAM')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-file', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# parser.add_argument('--sparsity', type=float, default=0.05, help="gradient sparsity")
parser.add_argument('--all_reduce', action='store_true', help='Using all_reduce')
parser.add_argument('--signum', action='store_true', help='Using Signum')
parser.add_argument('--compress', action='store_true', help='Initiate compression for Signum')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--gpus_per_machine", default=1, type=int)
parser.add_argument('--save-dir', type=str, default="./saving/", help='Directory to save logs and models.')

#LARC setting
parser.add_argument('--larc_enable', action='store_true', help='Enable the LARC')
parser.add_argument('--lamb_enable', action='store_true', help='Enable the LAMB')
parser.add_argument('--lambda0', type=float, default=0.01, help="scaling factor in LAMB")
parser.add_argument("--larc_trust_coefficient", default=0.02, type=float)
parser.add_argument("--larc_eps", default=1e-8, type=float)
parser.add_argument('--larc_clip', action='store_true', help='Enable the elip for LARC')

best_acc1 = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

class Time_recorder(object):
    def __init__(self):
        self.time = 0

    def reset(self):
        self.time = 0

    def set(self):
        torch.cuda.synchronize()
        self.begin = time.time()

    def record(self):
        torch.cuda.synchronize()
        self.end = time.time()
        self.time += self.end - self.begin

    def get_time(self):
        return self.time

def get_loaders(traindir, valdir, args, use_val_sampler=False, min_scale=0.08, Data_augmentation = True, split_data = False, seed = None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if Data_augmentation:
        train_dataset = Imagefolder_train_val.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), seed = seed, split = split_data, train_data = True)
    else:
        train_dataset = Imagefolder_train_val.ImageFolder(
            traindir,
            transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]), seed = seed, split = split_data, train_data = True)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if split_data:
        val_loader = torch.utils.data.DataLoader(
            Imagefolder_train_val.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), seed = seed, split = split_data, train_data = False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader,val_loader,train_sampler 

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # slurm available
    import os
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "distfiles/dist_url." + jobid  + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            # with open(hostfile, "w") as f:
            #     f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
        #     with open(hostfile, "r") as f:
        #         args.dist_url = f.read()
        # print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    
    # create model
    if args.dataset == "mnist":
        model = Net()
    elif args.dataset == "imagenet":
        model = models.resnet50()
    elif args.dataset == "tinyimagenet":
        # model = models.resnet50()
        model = ResNet18()
    elif args.dataset == "cifar":
        # model = models.resnet18()
        model = ResNet18()

    # Prepare Logger file
    if args.dataset == "mnist":
        logname = "vanilla"
    elif args.dataset == "imagenet":
        logname = "resnet50"
    elif args.dataset == "tinyimagenet":
        logname = "resnet18"
    elif args.dataset == "cifar":
        logname = "resnet18"
    
    print(colored("Create model {} for {} dataset".format(logname, args.dataset), 'green'))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('resnet'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(colored("Create Loss function", 'green'))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    log_writer = None
    print(colored("Init {} optimizer".format(args.optimizer), 'green'))
    if args.optimizer == "fedlamb":
        args.lamb_enable = True
        optimizer = FedLAMB(model.parameters(), args, log_writer)
    elif args.optimizer == "fedams":
        args.lamb_enable = False
        optimizer = FedLAMB(model.parameters(), args, log_writer)
    elif args.optimizer == "fedlars":
        args.larc_enable = True
        optimizer = SGD_distribute(model.parameters(), args, log_writer)
    elif args.optimizer == "fedsgd":
        args.larc_enable = False
        optimizer = SGD_distribute(model.parameters(), args, log_writer)

    print(colored("Create Logger file"), 'green')
    #logger file
    dataset = args.dataset
    title = '{}-{}'.format(dataset, logname)
    checkpoint_dir = 'checkpoints/checkpoints_{}'.format(dataset)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        pass
    
    if args.optimizer in ["fedlamb", "fedlars"]:
        larc = True
    else:
        larc = False

    logger = Logger('{}/{}_opt{}_LAMB{}_lambda{}_workers{}_lr{}_epoch{}.txt'.format(checkpoint_dir, logname, args.optimizer, larc, args.lambda0, args.workers, args.lr, args.epochs), title=title)
    # logger.set_names(['Learning Rate', 'Loss','Acc. at 1','Acc. at 5', 'Time'])
    logger.set_names(['Learning Rate','Loss','Acc. at 1','Acc. at 5', 'Time'])


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print(colored("Init {} dataset".format(args.dataset), 'green'))
    # Data loading code
    if args.dataset == "mnist":
        train_dataset = datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_dataset = datasets.CIFAR10('./data', train=True, download=True,transform=transform_train)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
            
    elif args.dataset == "imagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        args.sz = 224
        train_loader,val_loader,train_sampler = get_loaders(traindir, valdir,args, split_data = not args.test_evaluate, seed = args.seed)
    
    elif args.dataset == "tinyimagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        args.sz = 224
        train_loader,val_loader,train_sampler = get_loaders(traindir, valdir,args, split_data = not args.test_evaluate, seed = args.seed)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    start_time_cust = time.time()

    print(colored("Enter Training loop", 'green'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        elapsed_time = time.time() - start_time_cust
        logger.append([args.lr, loss, best_acc1, acc5,elapsed_time])


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return loss.item()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()