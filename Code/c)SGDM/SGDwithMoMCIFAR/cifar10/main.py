'''Train CIFAR10 with PyTorch.'''
#from __future__ import print_function

import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('../')
import pytorch_optim as optim_local
import os
import argparse
from models import *
#from progress import progress_bar
from logger import Logger, savefig
import numpy as np

# Input argument parsing
parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
# Model
parser.add_argument('--dataset', default='cifar10', type=str, choices=['mnist', 'cifar10'], help='Dataset')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', nargs='+', default=[80, 120], help='Either str diagnostic / plateau or epochs to decrease learning rate at.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--final_momentum', default=0.1, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--model', '-m', default='ResNet18', choices=['ShuffleNetV2', 'ResNet18', 'ResNeXt29', 'MnistNet', 'MnistNetSmall', 'MnistNetLarge'],
                    help='which CNN model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--optimizer', default='signSGD', type=str, choices=['signSGD', 'SGDM', 'ADAM'], help='optimization algorithm')
parser.add_argument('--beta2', default=0.999, type=float, help='betar2 EMA var scale for ADAM')

# Runtime
parser.add_argument('--gpu', '-g', type=int, nargs='+', default=[0], help='gpu index') #choices=['0','1','2','3','4','5','6','7'],
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--logname', default=False, type=str, help='name for log files, plots')
parser.add_argument('--loglevel', default='epoch', type=str, choices=['epoch', 'batch'], help='epoch or batch level saving or runtime stats')
parser.add_argument('--debug', default=False, type=bool, help='Debug mode. eliminates progress bar incompatible pycharm')

# Diagnostic
parser.add_argument('--burnin', default=10, type=int, help='burnin (epochs)')
parser.add_argument('--window', default=10, type=int, help='stationary test window (epochs)')
parser.add_argument('--sim', default='ip', choices=['ip', 'cosine'], help='inner prod. similarity')
parser.add_argument('--num_reduce', default=1, type=int, help='if diagnostic, number of times to reduce LR')
parser.add_argument('--momentum_switch', default=False, type=bool, help='Momentum reduction boolean')
parser.add_argument('--early_threshold', default=0.2, type=float, help='threshold for norm-based momentum switch')

# Create parser and extra variables
args = parser.parse_args()
if not args.debug:
    from progress import progress_bar
lr = args.lr
momentum = args.momentum
logname = args.logname
num_reduce = args.num_reduce
if 'diagnostic' not in args.schedule:
    args.schedule = [int(x) for x in args.schedule]
elif len(args.schedule) > 1:
    num_reduce = int(args.schedule[1])
momentum_ind = -1 # no momentum reduction
if args.momentum_switch:
    momentum_ind = 1 # look for momentum reduction point
device ='cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print(args.momentum)

# Data
print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

elif args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))

    ])
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#cifar10
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model = VGG('VGG19') ResNet18() PreActResNet18() GoogLeNet() DenseNet121() ResNeXt29_2x64d() MobileNet() MobileNetV2() DPN92()
#         ShuffleNetG2() SENet18() ShuffleNetV2(1.0) MnistNet()
print('==> Building model..')
if args.model == 'ResNet18':
    net = ResNet18()
elif args.model == 'ShuffleNetV2':
    net = ShuffleNetV2(1.0)
elif args.model == 'ResNeXt29':
    net = ResNeXt29_2x64d()
elif args.model == 'MnistNet':
    net = MnistNet()
elif args.model == 'MnistNetSmall':
    net = MnistNetSmall()
elif args.model == 'MnistNetLarge':
    net = MnistNetLarge()
net = net.to(device)
if 'cuda' in device:
    net = torch.nn.DataParallel(net, device_ids=args.gpu) #sometimes slower to parallelize
    cudnn.benchmark = True

# Logger
if (logname == False):
    logname = args.model
title = '{}-{}'.format(args.dataset, logname)
checkpoint_dir = 'checkpoint_{}'.format(args.dataset)
# Resume logger from checkpoint
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./{}/ckpt.{}'.format(checkpoint_dir, logname))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    logger = Logger('./checkpoint/log{}.txt'.format(logname), title=title, resume=True)
# New logger
else:
    logger = Logger('./{}/log{}.txt'.format(checkpoint_dir, logname), title=title)
    logger.set_names(['Learning Rate', 'Momentum', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.',
                      'IP Sum', 'IP Mean', 'IP Std', 'Grad Norm'])

# Optimzier, default signSGD
criterion = nn.CrossEntropyLoss()
if args.optimizer =='signSGD':
    optimizer = optim_local.sign_SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, sim=args.sim)
elif args.optimizer == 'SGDM':
    optimizer = optim_local.SGD_Diagnostic(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, sim=args.sim)
elif args.optimizer == 'ADAM':
    optimizer = optim_local.Adam_Diagnostic(net.parameters(), lr=1e-3, betas=(args.momentum, args.beta2), eps=1e-8, weight_decay=0, amsgrad=False)
if 'plateau' in args.schedule:
    scheduler = optim_local.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.burnin)

# Training
grad_norm = []
train_loss_ls = []
test_stat = 0.0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0; global train_loss_ls
    correct = 0
    total = 0
    ip_loss = []; global test_stat
    grad_loss = []; global grad_norm
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        _, diag_args = optimizer.step()
        ip_loss.append(diag_args['ip_loss'])
        grad_loss.append(diag_args['grad_loss'])

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if not args.debug:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | IP_sum: %.3f'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, np.sum(ip_loss)))

    # convergence tests based on inner product of loss list from epoch
    diag_stats = {'ip_loss_sum':np.sum(ip_loss), 'ip_loss_mean':np.mean(ip_loss), 'ip_loss_std':np.std(ip_loss),
                  'grad_norm_mean':np.mean(grad_loss)}
    grad_norm.append(diag_stats['grad_norm_mean'])
    train_loss_ls.append(train_loss)

    if (args.momentum_switch and momentum_ind == -1) or (not args.momentum_switch and epoch > args.burnin):
        test_stat += np.sum(ip_loss)

    return (train_loss, 100.*correct/total, diag_stats)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if not args.debug:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | test stat: %.3f'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, test_stat))

    return (test_loss, 100.*correct/total)

def save_checkpoint(state, test_acc):
    # Save checkpoint.
    global best_acc
    if test_acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './{}/ckpt.{}'.format(checkpoint_dir, logname))
        best_acc = test_acc

def adjust_learning_rate(optimizer, epoch, diag_stats):#, grad_norm, ip_loss):
    global lr
    global momentum
    global num_reduce
    global momentum_ind

    def change_momentum(m_new, opt):
        '''toggles momentum indicator and changes momentum parameter'''
        global momentum
        global momentum_ind

        opt.change_momentum(m_new)
        momentum_ind *= -1
        momentum = m_new
        print('Momentum change from %.2f to %.2f after epoch %d' % (args.momentum, args.final_momentum, epoch))

    # Momentun Reduction
    if args.momentum_switch and epoch > 0 and momentum_ind == 1 and \
        np.abs(train_loss_ls[epoch]-train_loss_ls[epoch-1])/train_loss_ls[epoch-1] < args.early_threshold:
        #np.abs(grad_norm[epoch]-grad_norm[epoch-1])/grad_norm[epoch-1] < args.early_threshold:
        change_momentum(args.final_momentum, optimizer)

    if 'plateau' in args.schedule and num_reduce >= 1 and momentum_ind == -1:
        if scheduler.step(diag_stats['ip_loss_sum']):
            lr *= args.gamma
            num_reduce -= 1
            change_momentum(args.momentum, optimizer)
        # if diag_stats['adf_convg'] and epoch > burnin:
        #     lr *= args.gamma
        #     burnin = epoch + args.burnin
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    elif 'diagnostic' in args.schedule and num_reduce >= 1 and momentum_ind == -1 and epoch > args.burnin:
        if np.sum(test_stat) < 0.0:
            print('LR reduce from %.2f to %.2f at epoch %d' % (lr, lr*args.gamma, epoch))
            lr *= args.gamma
            num_reduce -= 1
            change_momentum(args.momentum, optimizer)
    elif (epoch+1) in args.schedule and args.momentum_switch:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        change_momentum(args.momentum, optimizer)
        momentum_ind = 1 #only matters for momentum_switch=True case

for epoch in range(start_epoch, start_epoch+args.epochs):
    train_loss, train_acc, diag_stats = train(epoch)
    test_loss, test_acc = test(epoch)

    # append logger file
    logger.append([lr, momentum, train_loss, test_loss, train_acc, test_acc,
                   #diag_stats['ip_loss_sum']
                   test_stat, diag_stats['ip_loss_mean'], diag_stats['ip_loss_std'], diag_stats['grad_norm_mean']])

    adjust_learning_rate(optimizer, epoch, diag_stats)

    save_checkpoint({
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
    }, test_acc)

logger.close()
logger.plot(names=['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'IP Sum'],
            scaling='normalized')
savefig("./{}/log{}.eps".format(checkpoint_dir, logname))

print('Best acc:{}'.format(best_acc))

