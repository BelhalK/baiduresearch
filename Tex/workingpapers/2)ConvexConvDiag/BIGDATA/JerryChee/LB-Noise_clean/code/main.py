from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np

import sgd
from logger import Logger, savefig
from resnet import ResNet18
#from utils import progress_bar

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, loss_fn, eta, batch_mult=1):
    start = (epoch-1) * len(train_loader)
    model.train()
    print(eta/(1+start+0)**args.gamma)
    train_loss = 0; correct = 0; total = 0

    def batch_cnt(batch_idx):
        if (batch_mult == 1):
            return(batch_idx)
        elif (batch_mult == 2):
            return( (batch_idx+1)/2 )
        elif (batch_mult == 3):
            return( (batch_idx+2) /3 )
        else:
            return(batch_idx)

    # Simulated Large Batch
    count = 0
    grad_norm_ls = []; noise_record_ls = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if (count == 0 or batch_idx == len(train_loader)-1):
            optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target) / batch_mult
        loss.backward()
        if (count == 0 or batch_idx == len(train_loader)-1):
            _, grad_norm = optimizer.step(var=eta/( 1 + start + batch_cnt(batch_idx) )**args.gamma)
            count = batch_mult
        count -= 1
        grad_norm_ls.append(grad_norm)
        noise_record_ls.append(eta/( 1 + start + batch_cnt(batch_idx) )**args.gamma)

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    return(train_loss, 100.*correct/total, np.mean(grad_norm_ls), np.mean(noise_record_ls))


def train_bkup(args, model, device, train_loader, optimizer, epoch, loss_fn):
    start = (epoch-1) * len(train_loader)
    model.train()
    print(args.eta/(1+start+0)**args.gamma)
    train_loss = 0; correct = 0; total = 0
    grad_norm = 0; noise_record = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step(var=args.eta/(1+start+batch_idx)**args.gamma)

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    return(train_loss, 100.*correct/total, grad_norm, noise_record)

def test(args, model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if (args.dataset == 'mnist'):
                test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            elif (args.dataset == 'cifar10'):
                test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return(test_loss, 100. * correct/len(test_loader.dataset))

def update_lr_eta(optimizer, epoch, args, lr, eta):
    if epoch in args.schedule:
        lr = lr * 0.1
        eta = eta * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return(lr, eta, optimizer)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--grad-noise', type=bool, default=False, metavar='GN',
                        help='Add gradient noise to SGD (default: False)')
    parser.add_argument('--eta', type=float, default=0.3, metavar='eta',
                        help='Numerator coeff for variance reduction of add gaussian noise (default: 0.3)')
    parser.add_argument('--gamma', type=float, default=0.55, metavar='gamma',
                        help='Denominator power for variance reduction of add gaussian noise (default: 0.55)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gpu', '-g', type=int, nargs='+', default=[0],
                        help='gpu index')  # choices=['0','1','2','3','4','5','6','7'],
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--logname', type=str, default=False,
                        help='name for log files, plots')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'], help='Dataset')
    parser.add_argument('--batch-mult', '-bm', type=int, default=1,
                        help='Batch Multiplier for CIFAR10, for synthetic large batch size') # if want B=16,384, set batch-size=8,192 and batch-mult=2. Due to mem limit
    parser.add_argument('--schedule', type=int, nargs='+', default=[80,120], help='When to decrease LR and noise scale')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if (args.batch_size % args.batch_mult != 0):
        ValueError('Batch size and batch mult not divisible')
    lr = args.lr
    eta = args.eta

    torch.manual_seed(args.seed)

    device = 'cuda:{}'.format(args.gpu[0]) if use_cuda else 'cpu'

    # Dataset & model
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if (args.dataset == 'mnist'):
        Net = MnistNet()
        loss_fn = F.nll_loss
        netname = 'MnistNet'
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif (args.dataset == 'cifar10'):
        Net = ResNet18()
        loss_fn = nn.CrossEntropyLoss()
        netname = 'ResNet18'
        args.momentum = 0.9
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Logger
    logname = args.logname
    if (logname == False):
        logname ='{}-{}-E{}-BS{}-Mom{}-LR{}-eta{}-gamma{}'.format(args.dataset, netname, args.epochs, args.batch_size*args.batch_mult, args.momentum, lr, eta, args.gamma)
    title = '{}-{}'.format(args.dataset, logname)
    logger = Logger('./checkpoint/log{}.txt'.format(logname), title=title)
    logger.set_names(['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Grad Norm', 'Noise Record'])


    model = Net.to(device)
    if 'cuda' in device:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)  # sometimes slower to parallelize
        cudnn.benchmark = True
    optimizer = sgd.SGD(model.parameters(), lr=lr, momentum=args.momentum, grad_noise=args.grad_noise)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, grad_norm, noise_record = train(args, model, device, train_loader, optimizer, epoch, loss_fn, eta, args.batch_mult)
        test_loss, test_acc = test(args, model, device, test_loader, loss_fn)
        lr, eta, optimizer = update_lr_eta(optimizer, epoch, args, lr, eta)
        print(lr)
        print(eta)

        logger.append([train_loss, test_loss, train_acc, test_acc, grad_norm, noise_record])

    logger.close()
    logger.plot(names=['Valid Loss', 'Train Loss', 'Train Acc.', 'Valid Acc.'])
    savefig("./checkpoint/log{}.eps".format(logname))

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
