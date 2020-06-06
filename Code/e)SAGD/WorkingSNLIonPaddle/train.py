import os
import math
import pickle
import torch
import torch.optim as optimizers

from optim.SAGD import SAGD
from optim.SAGD_sparse import SAGDSparse
from optim.adabound import AdaBound

import torch.nn as nn
import numpy as np
import datasets
import models
import datetime
import torch.nn.functional as F
from tqdm import tqdm
from prettytable import PrettyTable
from utils import get_args, get_device
import time

class Train():
	def __init__(self):
		print("program execution start: {}".format(datetime.datetime.now()))
		self.args = get_args("train")
		self.device = get_device(self.args.gpu)
		self.dataset_options = {
									'batch_size': self.args.batch_size, 
									'device': self.device
								}
		self.dataset = datasets.__dict__[self.args.dataset](self.dataset_options)
		self.model_options = {
									'vocab_size': self.dataset.vocab_size(), 
									'embed_dim': self.args.embed_dim,
									'out_dim': self.dataset.out_dim(),
									'dp_ratio': self.args.dp_ratio,
									'd_hidden': self.args.d_hidden
								}
		self.model = models.__dict__[self.args.model](self.model_options)
		self.model.to(self.device)
		self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
		noi = np.log(self.args.batch_size) / self.args.batch_size
		if self.args.optimizer == 'sgd':
			self.opt = optimizers.SGD(self.model.parameters(), lr = self.args.lr, momentum=self.args.momentum,weight_decay=self.args.wdecay)
		elif self.args.optimizer == 'adagrad':
			self.opt = optimizers.Adagrad(self.model.parameters(), self.args.lr, weight_decay=self.args.wdecay)
		elif self.args.optimizer == 'adam':
			self.opt = optimizers.Adam(self.model.parameters(), lr = self.args.lr, betas=(self.args.beta1, self.args.beta2),weight_decay=self.args.wdecay)
		elif self.args.optimizer == 'amsgrad':
			self.opt = optimizers.Adam(self.model.parameters(), self.args.lr, betas=(self.args.beta1, self.args.beta2),weight_decay=self.args.wdecay, amsgrad=True)
		elif self.args.optimizer == 'adabound':
			self.opt = AdaBound(self.model.parameters(), lr = self.args.lr,betas=(self.args.beta1, self.args.beta2),final_lr=self.args.final_lr, gamma=self.args.gamma,weight_decay=self.args.wdecay)
		elif self.args.optimizer == 'sagd':
			self.opt = SAGD(self.model.parameters(), lr=self.args.lr, noise=self.args.noise_coe * noi, momentum=self.args.momentum, weight_decay=self.args.wdecay)

		self.best_accuracy = -1
		print("resource preparation done: {}".format(datetime.datetime.now()))

	def save_model(self, current_accuracy, repeat_id):
		if current_accuracy > self.best_accuracy:
			self.best_accuracy = current_accuracy
			torch.save({
				'accuracy': self.best_accuracy,
				'options': self.model_options,
				'model_dict': self.model.state_dict(),
			}, 'save/' + "{}-{}-repeat-{}-{}.pt".format(self.args.dataset, self.args.model, repeat_id,self.args.optimizer))
		pass
	
	def train(self):
		self.model.train(); self.dataset.train_iter.init_epoch()
		n_correct, n_total, n_loss = 0, 0, 0
		ind = 0
		for batch_idx, batch in enumerate(self.dataset.train_iter):
			while ind<2:
				self.opt.zero_grad()
				answer = self.model(batch)
				loss = self.criterion(answer, batch.label)
				n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
				n_total += batch.batch_size
				n_loss += loss.item()
				loss.backward(); self.opt.step()
				ind+= 1
				print(ind)
		train_loss = n_loss/n_total
		train_acc = 100. * n_correct/n_total
		return train_loss, train_acc

	def validate(self):
		self.model.eval(); self.dataset.dev_iter.init_epoch()
		n_correct, n_total, n_loss = 0, 0, 0
		with torch.no_grad():
			ind = 0
			for batch_idx, batch in enumerate(self.dataset.dev_iter):
				while ind<2:
					answer = self.model(batch)
					loss = self.criterion(answer, batch.label)
					n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
					n_total += batch.batch_size
					n_loss += loss.item()
					ind+= 1
					print(ind)
			val_loss = n_loss/n_total
			val_acc = 100. * n_correct/n_total
			return val_loss, val_acc

	def execute(self,repeat_idx):
		all_train_loss = []
		all_train_acc = []
		all_test_loss = []
		all_test_acc = []
		save_path = self.args.save + "." + self.args.optimizer + ".lr" + str(self.args.lr) + ".repeat" + str(repeat_idx)
		for epoch in range(self.args.epochs):
			start = time.time()
			train_loss, train_acc = self.train()
			val_loss, val_acc = self.validate()
			all_train_loss.append(train_loss)
			all_train_acc.append(train_acc)
			all_test_loss.append(val_loss)
			all_test_acc.append(val_acc)
			if self.args.save_model:
				self.save_model(val_acc, repeat_idx)
			print("time taken: {}   epoch: {}   Training loss: {}   Training Accuracy: {}   Validation loss: {}   Validation Accuracy: {}".format(
				round(time.time()-start, 2), epoch, round(train_loss, 3), round(train_acc, 3), round(val_loss, 3), round(val_acc, 3)
			))
		print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(val_loss, math.exp(val_loss), val_loss / math.log(2)))
		return all_train_loss, all_test_loss, all_train_acc, all_test_acc

def get_ppl(loss):
    if isinstance(loss, list):
        ppl = [get_ppl(a) for a in loss]
        return ppl
    else:
        return math.exp(min(loss, 10))

def set_random_seed(seed, cuda):
    """
    set random seed
    """
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)
def main():
	task = Train()
	repeat_all_train_loss = []
	repeat_all_train_acc = []
	repeat_all_test_loss = []
	repeat_all_test_acc = []
	for repeat_idx in range(task.args.repeat):
		print('start running {} repeat'.format(repeat_idx))
		set_random_seed(task.args.seed + repeat_idx, task.args.cuda)
		all_train_loss, all_test_loss, all_train_acc, all_test_acc = task.execute(repeat_idx)
		repeat_all_train_loss.append(all_train_loss)
		repeat_all_train_acc.append(all_train_acc)
		repeat_all_test_loss.append(all_test_loss)
		repeat_all_test_acc.append(all_test_acc)
	repeat_all_train_ppl = get_ppl(repeat_all_train_loss)
	repeat_all_test_ppl = get_ppl(repeat_all_test_loss)
	
	result = {
        "train_loss": repeat_all_train_loss,
        "train_ppl": repeat_all_train_ppl,
        "all_test_loss": repeat_all_test_loss,
        "all_test_ppl": repeat_all_test_ppl,
		"all_test_acc": repeat_all_test_acc}
	file_path = "lr{}_no{}_b{}_m{}_{}".format(task.args.lr, task.args.noise_coe, task.args.batch_size, task.args.momentum, task.args.optimizer)
	file_name = 'result_{}_b{}'.format(task.args.optimizer, task.args.batch_size)
	file_dir = os.path.join('output', task.args.model, file_name)
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	
	file_path = os.path.join(file_dir, file_path)
	with open(file_path, 'wb') as fou:
		pickle.dump(result, fou)


if __name__ == "__main__":
	main()