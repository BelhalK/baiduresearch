import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from models import *
import torch.backends.cudnn as cudnn
import pickle

from optimizers.adabound import AdaBound
from optimizers.Padam import Padam
from optimizers.SAGD import SAGD
from optimizers.SAdagrad import SAdagrad
from optimizers.SARMSprop import SARMSprop
from optimizers.SAGD_sparse import SAGDSparse
from optimizers.SAdagrad_sparse import SAdagradSparse
from optimizers.SARMSprop_sparse import SARMSpropSparse

# from datasets import CIFAR10
from datasets import SubData



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vggnet'])
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--noise-coe', type=float, default=1, metavar='NO',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-id', type=str, default='0',
                        help='the cuda id device')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--repeat', type=int, default=1,
                        help='number of repeated trainings')
    parser.add_argument('--LR-decay',  default= 'False',
                        help='Decay learning rate by epoch', choices=['False', 'True'])
    parser.add_argument('--decay-epoch', type=int, default=80, metavar='N',
                        help='number of epochs to decay (default: 10)')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--optim', default='adagrad', type=str, help='optimizer',
                        choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'padam','amsbound', 'rmsprop',
                        'sagd', 'sagd_sparse', 'SAdagrad', 'SAdagrad_sparse', 'SARMSprop', 'SARMSprop_sparse'])
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    return parser


def build_dataset(args, return_transform=False):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transform_test)
    if return_transform:
        return trainset, testset, transform_train, transform_test
    else:
        return trainset, testset


def create_optimizer(args, model_params):
    noi = np.log(args.batch_size) / args.batch_size
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'padam':
        return Padam(model_params, lr=args.lr, partial = 0.125, weight_decay = args.weight_decay, betas = (args.beta1, args.beta2))

    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    elif args.optim == 'amsbound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay, amsbound=True)
    elif args.optim == 'sagd':
        return SAGD(model_params, lr=args.lr, noise=args.noise_coe * noi, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == "sagd_sparse":
        return SAGDSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'SAdagrad':
        return SAdagrad(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.weight_decay)
    elif args.optim == 'SAdagrad_sparse':
        return SAdagradSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.weight_decay)
    elif args.optim == 'SARMSprop':
        return SARMSprop(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.weight_decay)
    elif args.optim == 'SARMSprop_sparse':
        return SARMSpropSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.weight_decay)
    elif args.optim == "rmsprop":
        return optim.RMSprop(model_params, lr=args.lr, weight_decay=args.weight_decay)

def build_model(args, device):
    print('==> Building model..')
    if args.model == 'vggnet':
#        from models import vgg
        model = VGG('VGG19')
#     model = models.vgg16_bn(num_classes=10)
    elif args.model == 'resnet':
#        from models import resnet
        model = ResNet18()
#     model = models.resnet18(num_classes=10)
    elif args.model == 'densenet':
#        from models import densenet
        model = DenseNet121()
    else:
        print ('Network undefined!')

#    model = model().to(device)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
#    if device == 'cuda':
#        model.cuda()
#        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
#        cudnn.benchmark = True
    return model



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    train_loss  = train_loss/(batch_idx+1)
    train_acc = 100.*correct/total

    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))
    return train_loss, train_acc


def train_sparse(args, model, device, train_loader, optimizer, epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        split = int(data.size()[0]/2)
        data1, target1 =  data[0:split], target[0:split]
        data2, target2 =  data[split:], target[split:]


        output1 = model(data1)
        output2 = model(data2)

        criterion = nn.CrossEntropyLoss()

        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)

        optimizer.zero_grad()
        loss1.backward()
        grad1= optimizer.get_grad()

        optimizer.zero_grad()
        loss2.backward()
#        grads2 = optimizer.get_grad()

        optimizer.step(grad1)

#        train_loss += loss.item()
#        _, predicted = output.max(1)
#        total += target.size(0)
#        correct += predicted.eq(target).sum().item()

#
#        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    train_loss  = train_loss/(batch_idx+1)
    train_acc = 100.*correct/total

    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))

    return train_loss, train_acc


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss = test_loss/(batch_idx+1)

    test_acc = 100.*correct/total
    print('\nTest loss: {}, Accuracy: {}'.format(test_loss, test_acc))
    return test_loss, test_acc



def main():
    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(0)

    print('run model {} on device cuda{}'.format(args.model, args.cuda_id))
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = 'cuda:' + args.cuda_id if use_cuda else 'cpu'
    train_set, test_set, train_transform, test_transform = build_dataset(
        args, True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size,
        shuffle=False, num_workers=10)


    train_set_sample = SubData(
        train_set.data, train_set.targets,
        transform=train_transform)

    """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True,
        num_workers=10)
    """
    def one_run(train_sample_num):
        #data_sums = [sample.sum().item() for sample in train_set.data]

        train_set_sample.set_sub_sample(train_sample_num)
        
        """
        sample_data_sums = [sample.sum().item() for sample in train_set_sample.data]
        """
        train_loader = torch.utils.data.DataLoader(
            train_set_sample, batch_size=args.batch_size,
            num_workers=10)

        model = build_model(args, device)
        optimizer  = create_optimizer(args, model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= args.decay_epoch, gamma=0.1,
                                              last_epoch=-1)

        Tr_loss = []
        Tr_acc = []
        Te_loss = []
        Te_acc = []
        for epoch in range(1, args.epochs + 1):

            if args.LR_decay == 'True':
                scheduler.step()

            if "sparse" in args.optim:
                train_loss, train_acc = train_sparse(args, model, device, train_loader, optimizer, epoch)
            else:
                train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(args, model, device, test_loader)
            Tr_loss.append(train_loss)
            Tr_acc.append(train_acc)
            Te_loss.append(test_loss)
            Te_acc.append(test_acc)
        return Tr_loss, Tr_acc, Te_loss, Te_acc


    train_sample_nums = [200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]

    all_train_accs = []
    all_test_accs = []

    all_train_losses = []
    all_test_losses = []

    for train_sample_num in train_sample_nums:
        Train_loss = []
        Train_acc = []
        Test_loss = []
        Test_acc = []
        for no_repeat in range(args.repeat):
            torch.manual_seed(no_repeat)
            Tr_loss, Tr_acc, Te_loss, Te_acc = one_run(train_sample_num)
            Train_loss.append(Tr_loss)
            Train_acc.append(Tr_acc)
            Test_loss.append(Te_loss)
            Test_acc.append(Te_acc)
        all_train_accs.append(Train_acc)
        all_test_accs.append(Test_acc)
        all_train_losses.append(Train_loss)
        all_test_losses.append(Test_loss)

    result = {
        "train_loss": all_train_losses, "train_acc": all_train_accs,
        "test_loss": all_test_losses, "test_acc":all_test_accs}


    file_path = "Decay_{}_lr{}_no{}_b{}_m{}_{}".format(
        args.LR_decay, args.lr, args.noise_coe, args.batch_size, args.momentum, args.optim)
    file_name = 'result_{}_b{}'.format(args.optim, args.batch_size)
#    os.mkdir(file_name)
    file_dir = os.path.join('output', args.model, file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_path)

    with open(file_path, 'wb') as fou:
        pickle.dump(result, fou)


if __name__ == '__main__':
    main()
