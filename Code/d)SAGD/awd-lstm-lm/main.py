import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
import model

from utils import batchify, get_batch, repackage_hidden
from utils import get_batch_for_sparse_optimizer
from optim.SAGD import SAGD
from optim.SAdagrad import SAdagrad
from optim.SARMSprop import SARMSprop
from optim.SAGD_sparse import SAGDSparse
from optim.SAdagrad_sparse import SAdagradSparse
from optim.SARMSprop_sparse import SARMSpropSparse
from optim.adabound import AdaBound
from optim.Padam import Padam
from splitcross import SplitCrossEntropyLoss
import os
import hashlib
import pickle


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--disable_asgd', action='store_true',
                        help='disable asgd')
    parser.add_argument('--device', type=str,
                        help='the device to run model')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str,  default=os.path.join("output", randomhash+'.pt'),
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)',
                        choices=[
                            'sagd', 'SAdagrad', 'SARMSprop', 'sagd_sparse',
                            'SAdagrad_sparse', 'SARMSprop_sparse', 'sgd',
                            'adagrad', 'adam', 'amsgrad', 'adabound',
                            'padam','amsbound', 'RMSprop'])
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--repeat', type=int, default=3,
                        help='number of repeated trainings')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--noise-coe', type=float, default=1, metavar='NO',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument(
        '--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument(
        '--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    args = parser.parse_args()
    args.tied = True
    return args


def create_optimizer(args, model_params):
    noi = np.log(args.batch_size) / args.batch_size
    if args.optimizer == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.wdecay)
    elif args.optimizer == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wdecay)
    elif args.optimizer == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wdecay, amsgrad=True)
    elif args.optimizer == 'padam':
        return Padam(model_params, lr=args.lr, partial = 0.125, weight_decay = args.wdecay, betas = (args.beta1, args.beta2))

    elif args.optimizer == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.wdecay)
    elif args.optimizer == 'amsbound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.wdecay, amsbound=True)
    elif args.optimizer == 'sagd':
        return SAGD(model_params, lr=args.lr, noise=args.noise_coe * noi, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimizer == "sagd_sparse":
        return SAGDSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimizer == 'SAdagrad':
        return SAdagrad(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.wdecay)
    elif args.optimizer == 'SAdagrad_sparse':
        return SAdagradSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.wdecay)
    elif args.optimizer == 'SARMSprop':
        return SARMSprop(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.wdecay)
    elif args.optimizer == 'SARMSprop_sparse':
        return SARMSpropSparse(model_params, lr=args.lr, noise=args.noise_coe * noi, weight_decay=args.wdecay)
    elif args.optimizer == "RMSprop":
        return optim.RMSprop(model_params, lr=args.lr, weight_decay=args.wdecay)


def load_corpus(args):
    """
    load corpus
    """
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)
    return corpus


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


def model_save(fn, model, criterion, optimizer):
    """
    model save
    """
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    """
    load model
    """
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer


def build_model(ntokens, args):
    """
    build model
    """
    rnn = model.RNNModel(
        args.model, ntokens, args.emsize, args.nhid,
        args.nlayers, args.dropout, args.dropouth,
        args.dropouti, args.dropoute, args.wdrop, args.tied)
    
    splits = []
    if ntokens > 500000:
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        splits = [2800, 20000, 76000]
    criterion = SplitCrossEntropyLoss(
        args.emsize, splits=splits, verbose=False)
    return rnn, criterion


def evaluate(data_source, args, model, criterion, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        data, targets = data.to(args.device), targets.to(args.device)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(train_data, args, model, criterion, optimizer, params, epoch):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    all_loss = 0
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        data, targets = data.to(args.device), targets.to(args.device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data

        all_loss += raw_loss.item()
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return all_loss / batch


def train_sparse(train_data, args, model, criterion, optimizer, params, epoch):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    all_loss = 0
    total_loss = 0
    start_time = time.time()

    hidden1 = model.init_hidden(args.batch_size // 2)
    hidden2 = model.init_hidden(args.batch_size // 2)

    batch, i = 0, 0
    split = args.batch_size // 2

    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch_for_sparse_optimizer(
            train_data, i, args, seq_len=seq_len)
        data, targets = data.to(args.device), targets.to(args.device)

        data1, target1 =  data[:, 0:split], targets[:, 0:split]
        data2, target2 =  data[:, split:], targets[:, split:]

        target1 = target1.contiguous().view(-1)
        target2 = target2.contiguous().view(-1)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden1 = repackage_hidden(hidden1)
        hidden2 = repackage_hidden(hidden2)


        output1, hidden1, rnn_hs1, dropped_rnn_hs1 = model(data1, hidden1, return_h=True)
        raw_loss1 = criterion(model.decoder.weight, model.decoder.bias, output1, target1)

        loss1 = raw_loss1
        # Activiation Regularization
        if args.alpha: loss1 = loss1 + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs1[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss1 = loss1 + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs1[-1:])


        output2, hidden2, rnn_hs2, dropped_rnn_hs2 = model(data2, hidden2, return_h=True)
        raw_loss2 = criterion(model.decoder.weight, model.decoder.bias, output2, target2)
        loss2 = raw_loss2

        # Activiation Regularization
        if args.alpha: loss2 = loss2 + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs2[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss2 = loss2 + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs2[-1:])

        optimizer.zero_grad()
        loss1.backward()

        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        grad1 = optimizer.get_grad()

        optimizer.zero_grad()
        loss2.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step(grad1)

        total_loss += (raw_loss1.data + raw_loss2.data) / 2
        all_loss += (raw_loss1.item() + raw_loss2.item()) / 2

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return all_loss / batch


def get_ppl(loss):
    if isinstance(loss, list):
        ppl = [get_ppl(a) for a in loss]
        return ppl
    else:
        return math.exp(min(loss, 10))


def main():
    args = get_parser()
    corpus = load_corpus(args)
    eval_batch_size = 10
    test_batch_size = 1
    # train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ntokens = len(corpus.dictionary)

    train_total_num = corpus.train.size(0)
    print('train total num ', train_total_num)
    valid_total_num = corpus.valid.size(0)
    print('total valid num ', valid_total_num)
    test_total_num = corpus.test.size(0)
    print('total test num ', test_total_num)

    import pdb; pdb.set_trace()

    np.random.seed(args.seed)

    def one_run(train_sample_num=0, repeat_idx=0):
        """
        one run
        """
        sample_train = None
        if train_sample_num == 0:
            sample_train = corpus.train
        else:
            sampled_idx = np.random.randint(train_total_num - train_sample_num)
            sample_train = corpus.train[sampled_idx: sampled_idx + train_sample_num]

        train_data = batchify(sample_train, args.batch_size, args)
        model, criterion = build_model(ntokens, args)
        model = model.to(args.device)
        criterion = criterion.to(args.device)

        params = list(model.parameters()) + list(criterion.parameters())
        total_params = sum(
            x.size()[0] * x.size()[1] \
                if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
        print('Args:', args)
        print('Model total parameters:', total_params)


        # Loop over epochs.
        lr = args.lr
        best_val_loss = []
        stored_loss = 100000000

        # At any point you can hit Ctrl + C to break out of training early.
        optimizer = create_optimizer(args, params)

        all_train_loss = []
        all_valid_loss = []
        all_test_loss = []

        save_path = args.save + "." + args.optimizer + ".lr" + \
            str(args.lr) + ".sample" + str(train_sample_num) + \
                ".noise" + str(args.noise_coe) + \
                ".layer" + str(args.nlayers) + \
                ".repeat" + str(repeat_idx)

        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            if "sparse" in args.optimizer:
                train_loss = train_sparse(
                    train_data, args, model, criterion,
                    optimizer, params, epoch)
            else:
                train_loss = train(
                    train_data, args, model, criterion,
                    optimizer, params, epoch)
            
            """
            print('epoch train {} done, time {}'.format(
                epoch, time.time() - epoch_start_time))
            """

            all_train_loss.append(train_loss)

            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                    """
                    if 'ax' in  optimizer.state[prm]:
                        prm.data = optimizer.state[prm]['ax'].clone()
                    else:
                        print('ax not in state prm')
                        print('prm ', prm)
                    """

                val_loss2 = evaluate(
                    val_data, args, model, criterion, eval_batch_size)
                all_valid_loss.append(val_loss2)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(save_path, model, criterion, optimizer)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                #val_start_time = time.time()
                val_loss = evaluate(
                    val_data, args, model, criterion, eval_batch_size)
                #print('eoch {} val loss done, time {}'.format(
                #    epoch, time.time() - val_start_time))
                #test_start_time = time.time()
                test_loss = evaluate(
                    test_data, args, model, criterion, test_batch_size)
                #print('epoch {} test loss done, time {}'.format(
                #   epoch, time.time() - test_start_time))
                all_valid_loss.append(val_loss)
                all_test_loss.append(test_loss)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(save_path, model, criterion, optimizer)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                # args.optimizer == 'sgd' and
                if not args.disable_asgd and 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(
                        model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(
                        save_path, epoch), model, criterion, optimizer)
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

        model, criterion, optimizer = model_load(save_path)
        # Run on test data.
        test_loss = evaluate(test_data, args, model, criterion, test_batch_size)

        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)
        print('training using optimizer {} done'.format(args.optimizer))
        print('train loss {},  test loss {}'.format(train_loss, test_loss))
        return all_train_loss, all_valid_loss, all_test_loss, test_loss


    samples_all_train_loss = []
    samples_all_train_ppl = []


    samples_all_valid_loss = []
    samples_all_valid_ppl = []

    samples_all_test_loss = []
    samples_all_test_ppl = []

    samples_test_loss = []
    samples_test_ppl = []

    # train_sample_nums = [10000, 50000, 100000, 300000, 500000, 700000]
    train_sample_nums = [0]

    for train_sample_num in train_sample_nums:
        repeat_train_loss = []
        repeat_valid_loss = []
        repeat_all_test_loss = []
        repeat_test_loss = []

        repeat_num = args.repeat

        """
        if train_sample_num == 0:
            repeat_num = 1
        """

        for repeat_idx in range(repeat_num):
            print('start running {} repeat'.format(repeat_idx))
            set_random_seed(args.seed + repeat_idx, args.cuda)
            all_train_loss, all_valid_loss, all_test_loss, test_loss = one_run(
                train_sample_num, repeat_idx)
            repeat_train_loss.append(all_train_loss)
            repeat_valid_loss.append(all_valid_loss)
            repeat_all_test_loss.append(all_test_loss)
            repeat_test_loss.append(test_loss)
        
        samples_all_train_loss.append(repeat_train_loss)
        samples_all_valid_loss.append(repeat_valid_loss)
        samples_all_test_loss.append(repeat_all_test_loss)
        samples_test_loss.append(repeat_test_loss)
    
    samples_all_train_ppl = get_ppl(samples_all_train_loss)
    samples_all_valid_ppl = get_ppl(samples_all_valid_loss)
    samples_all_test_ppl = get_ppl(samples_all_test_loss)
    samples_test_ppl = get_ppl(samples_test_loss)

    result = {
        "train_loss": samples_all_train_loss,
        "train_ppl": samples_all_train_ppl,
        "valid_loss": samples_all_valid_loss,
        "valid_ppl": samples_all_valid_ppl,
        "all_test_loss": samples_all_test_loss,
        "all_test_ppl": samples_all_test_ppl,
        "test_loss": samples_test_loss, "test_ppl": samples_test_ppl}

    file_path = "lr{}_no{}_b{}_m{}_{}_asgd_{}".format(
        args.lr, args.noise_coe, args.batch_size, args.momentum, args.optimizer,
        not args.disable_asgd)
    file_name = 'result_{}_b{}'.format(args.optimizer, args.batch_size)
    file_dir = os.path.join('output', args.model, file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    file_path = os.path.join(file_dir, file_path)
    with open(file_path, 'wb') as fou:
        pickle.dump(result, fou)


if __name__ == "__main__":
    main()