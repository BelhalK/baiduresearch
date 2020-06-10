import torch
from argparse import ArgumentParser
import time
import os

def training_params():
	parser = ArgumentParser(description='PyTorch/torchtext NLI Tasks - Training')
	parser.add_argument('--dataset', type=str, default='snli')
	parser.add_argument('--model', type=str, default='bilstm')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--embed_dim', type=int, default=300)
	parser.add_argument('--d_hidden', type=int, default=512)
	parser.add_argument('--dp_ratio', type=int, default=0.2)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--combine', type=str, default='cat')
	parser.add_argument('--save_model', action='store_false', default=True)
	parser.add_argument('--optimizer', type=str,  default='sgd',help='optimizer to use (sgd, adam)',choices=['sagd', 'SAdagrad', 'SARMSprop', 'sagd_sparse','SAdagrad_sparse', 'SARMSprop_sparse', 'sgd','adagrad', 'adam', 'amsgrad', 'adabound','padam','amsbound', 'RMSprop'])
	parser.add_argument('--gamma', default=1e-3, type=float,help='convergence speed term of AdaBound')
	parser.add_argument('--noise-coe', type=float, default=1, metavar='NO',help='learning rate (default: 0.01)')
	parser.add_argument('--final_lr', default=0.1, type=float,help='final learning rate of AdaBound')
	parser.add_argument('--momentum', type=float, default=0, metavar='M',help='SGD momentum (default: 0)')
	parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
	parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
	parser.add_argument('--alpha', type=float, default=2,help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
	parser.add_argument('--beta', type=float, default=1,help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
	parser.add_argument('--wdecay', type=float, default=1.2e-6,help='weight decay applied to all weights')
	parser.add_argument('--resume', type=str,  default='',help='path of model to resume')
	parser.add_argument('--repeat', type=int, default=3,help='number of repeated trainings')
	randomhash = ''.join(str(time.time()).split('.'))
	parser.add_argument('--save', type=str,  default=os.path.join("output", randomhash+'.pt'),help='path to save the final model')
	parser.add_argument('--cuda', action='store_false',help='use CUDA')
	parser.add_argument('--seed', type=int, default=1111,help='random seed')
	args = parser.parse_args()
	return args

def evaluate_params():
	parser = ArgumentParser(description='PyTorch/torchtext NLI Tasks - Evaluation')
	parser.add_argument('--dataset', type=str, default='snli')
	parser.add_argument('--model', type=str, default='bilstm')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--save_path', type=str, default = "save/bilstm-snli-model.pt")
	args = parser.parse_args()
	return args

def get_args(mode):
	if mode == "train":
		return training_params()
	elif mode == "evaluate":
		return evaluate_params()

def get_device(gpu_no):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_no)
		return torch.device('cuda:{}'.format(gpu_no))
	else:
		return torch.device('cpu')