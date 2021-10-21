# python train_data_gen_flowers.py --th 0.0001 --eps 0.01 --mcmcmethod langevin
# python train_data_gen_flowers.py --th 0.0001 --eps 0.01 --mcmcmethod compressed

import torch
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
from nets import VanillaNet, NonlocalNet
from utils import download_flowers_data, plot_ims, plot_diagnostics, plot_single_ims

import argparse
from logger import Logger
import numpy as np
import compressor as sko
from customAdam import Adam
import collections
import pdb


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--th', '--thresh', default=0.001, type=float,
                    metavar='TH', help='threshold')
parser.add_argument('--eps', '--epsilon', default=0.001, type=float,
                    metavar='EP', help='epsilon')
parser.add_argument('--mcmcmethod', default='langevin',help='mcmc to use (langevin, compressed)')


args = parser.parse_args()


# json file with experiment config
CONFIG_FILE = './config_locker/flowers_nonconvergent.json' #flowers data
# CONFIG_FILE = './config_locker/mnist_convergent.json' #mnist data


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)


anith = args.th
eps = args.eps
mcmcmethod = args.mcmcmethod #can be compressed, langevin

# directory for experiment results
if mcmcmethod == "compressed":
    EXP_DIR = f'./out_data/flowers_nonconvergent_1_{mcmcmethod}_th{anith}_eps{eps}/'
else: 
    EXP_DIR = f'./out_data/flowers_nonconvergent_1_{mcmcmethod}_eps{eps}/'


# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    # raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
    pass
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)

title = 'rings-{}'.format(mcmcmethod)
if mcmcmethod=='compressed':
    logger = Logger('./out_data/log_{}_th{}_ep{}.txt'.format(mcmcmethod,anith,eps), title=title)
else:
    logger = Logger('./out_data/log_{}_ep{}.txt'.format(mcmcmethod,eps), title=title)
    
logger.set_names(['iteration','endiff','gradmag'])


# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_data.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
# save_code()

# set seed for cpu and CUDA, get device
torch.manual_seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################
# ## TRAINING SETUP # ##
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'vanilla': VanillaNet, 'nonlocal': NonlocalNet}
f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
# set up optimizer
# optim_bank = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
optim_bank = {'adam': Adam, 'sgd': torch.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
# make tensor of training data
if config['data'] == 'flowers':
    download_flowers_data()

data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}
transform = tr.Compose([tr.Resize(config['im_sz']),
                        tr.CenterCrop(config['im_sz']),
                        tr.ToTensor(),
                        tr.Normalize(tuple(0.5*torch.ones(config['im_ch'])), tuple(0.5*torch.ones(config['im_ch'])))])
q = torch.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

s_t_0 = 2 * torch.rand_like(q) - 1


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# sample batch from given array of images
def sample_image_set(image_set):
    rand_inds = torch.randperm(image_set.shape[0])[0:config['batch_size']]
    return image_set[rand_inds], rand_inds

# sample positive images from dataset distribution
def sample_q():
    x_q = sample_image_set(q)[0]
    return x_q + config['data_epsilon'] * torch.randn_like(x_q)

# initialize and update images with langevin dynamics
def sample_s_t(L, init_type, update_s_t_0=True):
    def sample_s_t_0():
        if init_type == 'persistent':
            return sample_image_set(s_t_0)
        elif init_type == 'data':
            return sample_q(), None
        elif init_type == 'uniform':
            noise_image = 2 * torch.rand([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]) - 1
            return noise_image.to(device), None
        elif init_type == 'gaussian':
            noise_image = torch.randn([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']])
            return noise_image.to(device), None
        else:
            raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    # initialize MCMC samples
    x_s_t_0, s_t_0_inds = sample_s_t_0()

    # iterative langevin updates of MCMC samples
    x_s_t = torch.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = torch.zeros(1).to(device)
    for ell in range(L):
        if mcmcmethod == "langevin":
            # regular langevin update constant LR
            f_prime = torch.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
            x_s_t.data += - f_prime + eps * torch.randn_like(x_s_t)
            r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()
        
        elif mcmcmethod == "compressed":
            # Gradient computation
            f_prime = torch.autograd.grad(f(x_s_t).sum(), [x_s_t])[0] 
            
            ### Compressed gradient
            new_f_prime = f_prime.view(f_prime.shape[0], -1)
            norm = torch.max(torch.abs(new_f_prime), dim=1, keepdim=True)[0]
            normalized_new_f_prime = new_f_prime / norm

            n_bit = 6
            sp = 2**n_bit
            scaled_vec = torch.abs(normalized_new_f_prime) * sp
            code_dtype = torch.int32
            lll = torch.clamp(scaled_vec, 0, sp-1).type(code_dtype)

            random = True
            if random:
                probabilities = scaled_vec - lll.type(torch.float32)
                rrr = torch.rand(lll.size())
                rrr = rrr.cuda()
                lll[:] += (probabilities > rrr).type(code_dtype)

            signs = torch.sign(new_f_prime) > 0
            pdb.set_trace()
            compvec = [norm, signs.view(new_f_prime.shape), lll.view(new_f_prime.shape)]
            
            # compvec = sko.compress(f_prime)

            ### Sketched gradient
            # t = 10
            # k = 50
            
            # hashtable = collections.OrderedDict()
            # signtable = collections.OrderedDict()

            # compvec = sko.generate_sketch(f_prime,t,k,hashtable,signtable)

            x_s_t.data += - comp_f_prime + eps * torch.randn_like(x_s_t)
            r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()


    if init_type == 'persistent' and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L



#######################
# ## TRAINING LOOP ## #
#######################

d_s_t_record = torch.zeros(config['num_train_iters']).to(device)  # energy difference between positive and negative samples
r_s_t_record = torch.zeros(config['num_train_iters']).to(device)  # average image gradient magnitude along Langevin path

print('Training has started.')
for i in range(config['num_train_iters']):
    print(i)
    # obtain positive and negative samples
    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'])
    
    # calculate loss
    d_s_t = f(x_q).mean() - f(x_s_t).mean()
    if config['epsilon'] > 0:
        # scale loss with the langevin implementation
        d_s_t *= 2 / (config['epsilon'] ** 2)
    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record[i] = d_s_t.detach().data
    r_s_t_record[i] = r_s_t

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # print and save learning info
    if (i + 1) == 1 or (i + 1) % config['log_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        logger.append([i+1, d_s_t.detach().data, r_s_t])
        
        # visualize synthesized images on a grid
        # plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_{:>06d}.png'.format(i+1), x_s_t)
        
        # visualize synthesized images and positive samples
        for index in range(len(x_s_t)):
            plot_single_ims(EXP_DIR + 'shortrun/' + '{:>06d}_x_s_t_0_{:>06d}.png'.format(i+1, index), x_s_t[index])
            plot_single_ims(EXP_DIR + 'shortrun/' + '{:>06d}_x_q_{:>06d}.png'.format(i+1, index), x_q[index])
        
        if config['shortrun_init'] == 'persistent':
            plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']])
        # save network weights
        torch.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i+1))
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, EXP_DIR + 'plots/')

    # sample longrun chains to diagnose model steady-state
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(L=config['num_longrun_steps'], init_type=config['longrun_init'], update_s_t_0=False)[0]
        plot_ims(EXP_DIR + 'longrun/' + 'longrun_{:>06d}.png'.format(i+1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i+1))
