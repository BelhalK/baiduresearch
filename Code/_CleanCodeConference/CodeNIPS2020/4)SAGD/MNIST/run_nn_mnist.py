import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import datasets
import torch.backends.cudnn as cudnn
import numpy as np
import os

#from SAGD import SAGD
from optimizers.adabound import AdaBound
from optimizers.Padam import Padam

from optimizers.SAGD import SAGD
from optimizers.SAdagrad import SAdagrad
from optimizers.SARMSprop import SARMSprop

from optimizers.SAGD_sparse import SAGDSparse
from optimizers.SAdagrad_sparse import SAdagradSparse
from optimizers.SARMSprop_sparse import SARMSpropSparse
import pickle


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--noise-coe', type=float, default=1, metavar='NO',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')


    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=[
                        'sagd', 'SAdagrad', 'SARMSprop', 'sagd_sparse', 'SAdagrad_sparse', 'SARMSprop_sparse',
                        'sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'padam','amsbound'])
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--repeat', type=int, default=1,
                        help='number of repeated trainings')
    parser.add_argument('--LR-decay',  default= 'False',
                        help='Decay learning rate by epoch', choices=['False', 'True'])
    parser.add_argument('--decay-epoch', type=int, default=5, metavar='N',
                        help='number of epochs to decay (default: 10)')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    return parser



def build_dataset(args):
    print('==> Preparing data..')
    train_set = datasets.MNIST(
        '../data', train=True, download=False,
        transform=transforms.ToTensor())
    test_set = datasets.MNIST(
        '../data', train=False, download=False,
        transform=transforms.ToTensor())
    
    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.SubMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
#                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), sample_num=train_sample_num),
        batch_size=args.batch_size, num_workers=20)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
 #                          transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, num_workers=20)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    '''
    return train_set, test_set

#
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.fc1 = nn.Linear(28 * 28, 256)
##        self.fc2 = nn.Linear(200, 200)
#        self.fc3 = nn.Linear(256, 10)
#
#    def forward(self, x):
#        x = F.relu(self.fc1(x))
##        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return F.log_softmax(x,dim=1)
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
    

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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28*28)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    train_loss = 0
    correct = 0
    with torch.no_grad():
        for  data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))
    return train_loss, train_acc


def train_sparse(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        split = int(data.size()[0]/2)
        data1, target1 =  data[0:split], target[0:split]
        data2, target2 =  data[split:], target[split:]

        data1 = data1.view(-1, 28*28)
        data2 = data2.view(-1, 28*28)

        output1 = model(data1)
        output2 = model(data2)

        loss1 = F.nll_loss(output1, target1)
        loss2 = F.nll_loss(output2, target2)

        optimizer.zero_grad()
        loss1.backward()
        grad1= optimizer.get_grad()

        optimizer.zero_grad()
        loss2.backward()

        optimizer.step(grad1)

    train_loss = 0
    correct = 0
    with torch.no_grad():
        for  data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))
    return train_loss, train_acc


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest loss: {}, Accuracy: {}'.format(test_loss, test_acc))
    return test_loss, test_acc

#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))


def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(0)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = 'cuda' if use_cuda else 'cpu'

    train_set, test_set = build_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size)
    train_set_sample = datasets.SubData(
        train_set.data, train_set.targets)

    def one_run(args, train_sample_num):
        # train_loader, test_loader = build_dataset(args, train_sample_num)
        train_set_sample.set_sub_sample(train_sample_num)

        train_loader = torch.utils.data.DataLoader(
            train_set_sample, batch_size=args.batch_size)
        model = Net().to(device)
        # model.zero_params()
#        train_loader, test_loader = build_dataset(args)

        optimizer  = create_optimizer(args, model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= args.decay_epoch, gamma=0.5,
                                              last_epoch=-1)

        Tr_loss = []
        Tr_acc = []
        Te_loss = []
        Te_acc = []

        for epoch in range(1, args.epochs + 1):

            print('epoch ', epoch)
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

        best_train_acc = max(Tr_acc)
        best_test_acc = max(Te_acc)
        best_train_loss = min(Tr_loss)
        best_test_loss = min(Te_loss)
        return best_train_acc, best_test_acc, best_train_loss, best_test_loss

    '''
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []
    '''


    train_sample_nums = [200, 500, 1000, 2000, 5000, 10000, 20000]

    all_train_accs = []
    all_test_accs = []

    all_train_losses = []
    all_test_losses = []

    for train_sample_num in train_sample_nums:
        train_accs = []
        test_accs = []
        train_loss = []
        test_loss = []

        for no_repeat in range(args.repeat):
            torch.manual_seed(no_repeat)

            print('train sample num {}, repeat num {}'.format(
                train_sample_num, no_repeat))
            best_train_acc, best_test_acc, best_train_loss, best_test_loss = one_run(args, train_sample_num)
            train_accs.append(best_train_acc)
            test_accs.append(best_test_acc)
            train_loss.append(best_train_loss)
            test_loss.append(best_test_loss)

        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)

    result = {'train_accs': all_train_accs, 'test_accs': all_test_accs,
              'train_losses': all_train_losses, 'test_losses': all_test_losses}

    file_path = "Decay_{}_lr{}_no{}_b{}_{}".format(args.LR_decay, args.lr, args.noise_coe, args.batch_size, args.optim)
    file_name = 'result_{}_b{}'.format(args.optim, args.batch_size)
#    os.mkdir(file_name)
    file_dir = os.path.join("output", file_name)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = os.path.join(file_dir, file_path)

    with open(file_path, 'wb') as fou:
        pickle.dump(result, fou)


if __name__ == '__main__':
    main()
