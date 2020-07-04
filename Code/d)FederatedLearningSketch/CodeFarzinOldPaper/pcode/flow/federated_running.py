# -*- coding: utf-8 -*-
import gc
import time
from copy import deepcopy
import numpy as np
import time

import torch
import torch.distributed as dist

from pcode.components.create_metrics import accuracy
from pcode.components.create_scheduler import adjust_learning_rate
from pcode.components.create_dataset import define_dataset, load_data_batch, _load_data_batch
from pcode.tracking.checkpoint import save_to_checkpoint
from pcode.flow.flow_utils import is_stop, get_current_epoch, get_current_local_step, update_client_epoch, zero_copy
from pcode.flow.communication import set_online_clients, distribute_model_server, fedavg_aggregation, global_average, distribute_model_server_control, \
                                    scaffold_aggregation, lgt_aggregation
from pcode.tracking.logging import log, logging_computing, logging_sync_time, \
    logging_display_training, logging_display_val, logging_load_time, \
    logging_globally, update_performancec_tracker
from pcode.tracking.meter import define_local_training_tracker,\
    define_val_tracker, evaluate_gloabl_performance, evaluate_local_performance

from pcode.components.create_optimizer import define_optimizer
from pcode.components.create_metrics import define_metrics


def inference(model, criterion, metrics, _input, _target):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics)
    return loss, performance

def do_test(args, model, optimizer, criterion, metrics, test_loader):
    """Evaluate the model on the test dataset."""
    performance = test(args, model, criterion, metrics, test_loader)

    # remember best prec@1 and save checkpoint.
    args.cur_prec1 = performance[0]
    is_best = args.cur_prec1 > args.best_prec1
    if is_best:
        args.best_prec1 = performance[0]
        args.best_epoch += [args.epoch_]

    # logging and display val info.
    logging_display_val(args)

    log('finished validation.', args.debug)


def test(args, model, criterion, metrics, test_loader):
    """A function for model evaluation."""
    # define stat.
    tracker = define_val_tracker()

    # switch to evaluation mode
    model.eval()
    log('Do validation on the server.', args.debug)
    for _input, _target in test_loader     :
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        with torch.no_grad():
            loss, performance = inference(
                model, criterion, metrics, _input, _target)
            tracker = update_performancec_tracker(
                tracker, loss, performance, _input.size(0))

    log('Aggregate test accuracy from different batches.', args.debug)
    performance = [
        evaluate_local_performance(tracker[x]) for x in ['top1', 'top5','losses']
    ]

    log('Test at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
        args.local_index, args.epoch, args.graph.rank, performance[0], performance[1], performance[2], args.rounds_comm),
        debug=args.debug)
    return performance

def do_validate(args, 
                model, 
                optimizer, 
                criterion, 
                metrics, 
                val_loader, 
                group,
                online_clients, 
                val=False):
    """Evaluate the model on the validation dataset."""

    tracker = define_val_tracker()
    
    # switch to evaluation mode
    model.eval()

    
    log('Do validation on the client models.', args.debug)
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        with torch.no_grad():
            loss, performance = inference(
                    model, criterion, metrics, _input, _target)
            tracker = update_performancec_tracker(
                tracker, loss, performance, _input.size(0))

    log('Aggregate val performance from different clients.', args.debug)
    performance = [
        evaluate_gloabl_performance(tracker[x], group) for x in ['top1', 'top5','losses']
    ]

    if val:
        log('Global performance for validation at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
            args.local_index, args.epoch, args.graph.rank, performance[0], performance[1], performance[2], args.rounds_comm),
            debug=args.debug)
    else:
        log('Global performance for train at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
            args.local_index, args.epoch, args.graph.rank, performance[0], performance[1], performance[2], args.rounds_comm),
            debug=args.debug)
    return performance


def train_and_validate_federated_complete(args, model_client, criterion, scheduler, optimizer, metrics):
    """The training scheme of Federated Learning systems.
        The basic model is FedAvg https://arxiv.org/abs/1602.05629
        TODO: Merge different models under this method
    """
    log('start training and validation with Federated setting.', args.debug)

    # get data loader.
    train_loader, test_loader = define_dataset(args, shuffle=True)

    args.finish_one_epoch = False
    all_clients_group = dist.new_group(args.graph.ranks)
    model_server = deepcopy(model_client)
    if args.federated_type in ['lgt','clgt']:
        model_delta = zero_copy(model_client)
    elif args.federated_type == 'scaffold':
        model_client_control = deepcopy(model_client)
        model_server_control = deepcopy(model_client)
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', args.debug)

    # Number of communication rounds in federated setting should be defined
    for n_c in range(args.num_comms):
        args.rounds_comm += 1
        args.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        # TODO: not make the server rank hard coded
        log("Starting round {} of training".format(n_c), args.debug)
        online_clients = set_online_clients(args)
        if n_c == 0:
            online_clients =  online_clients if 0 in online_clients else online_clients + [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if args.graph.rank in online_clients_server:
            if  args.federated_type == 'scaffold':
                st = time.time()
                model_server, model_server_control = distribute_model_server_control(model_server, model_server_control, online_clients_group, src=0)
                args.comm_time[-1] += time.time() - st
            else:
                st = time.time()
                model_server = distribute_model_server(model_server, online_clients_group, src=0)
                args.comm_time[-1] += time.time() - st
            model_client.load_state_dict(model_server.state_dict())
            local_steps = 0
            if args.graph.rank in online_clients:
                # for _ in range(args.num_epochs_per_comm):
                is_sync = False
                while not is_sync:
                    for _input, _target in train_loader:
                        local_steps += 1
                        model_client.train()

                        # update local step.
                        logging_load_time(tracker)

                        # update local index and get local step
                        args.local_index += 1
                        args.local_data_seen += len(_target)
                        get_current_epoch(args)
                        local_step = get_current_local_step(args)

                        # adjust learning rate (based on the # of accessed samples)
                        lr = adjust_learning_rate(args, optimizer, scheduler)

                        # load data
                        _input, _target = load_data_batch(args, _input, _target, tracker)
                        # inference and get current performance.
                        optimizer.zero_grad()
                        loss, performance = inference(model_client, criterion, metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss.backward()

                        if args.federated_type in ['lgt','clgt']:
                            # Update gradients with control variates
                            for client_param, delta_param  in zip(model_client.parameters(), model_delta.parameters()):
                                client_param.grad.data -= delta_param.data 
                        elif args.federated_type == 'scaffold':
                            for cp, ccp, scp  in zip(model_client.parameters(), model_client_control.parameters(), model_server_control.parameters()):
                                cp.grad.data += scp.data - ccp.data


                        optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=args.in_momentum, apply_out_momentum=False
                        )
                      
                        # logging locally.
                        # logging_computing(tracker, loss_v, performance_v, _input, lr)

                        
                        if args.epoch_ % 1 == 0:
                            args.finish_one_epoch = True
                        
                        # display the logging info.
                        # logging_display_training(args, tracker)

                        # finish one epoch training and to decide if we want to val our model.
                        if args.finish_one_epoch:
                            # each worker finish one epoch training.
                            if args.evaluate:
                                if args.epoch % args.eval_freq ==0:
                                    do_test(args, model_client, optimizer, criterion, metrics, test_loader)

                            # refresh the logging cache at the begining of each epoch.
                            args.finish_one_epoch = False
                            tracker = define_local_training_tracker()

                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
                        # model_local = deepcopy(model_client)
                        if args.federated_sync_type == 'local_step':
                            is_sync = args.local_index % local_step == 0
                        elif args.federated_sync_type == 'epoch':
                            is_sync = args.epoch_ % args.num_epochs_per_comm == 0
                        if is_sync:
                            break

            else:
                log("Offline in this round. Waiting on others to finish!", args.debug)

            # Sync the model server based on model_clients
            log('Enter synching', args.debug)
            tracker['start_sync_time'] = time.time()
            args.global_index += 1

            if args.federated_type == 'lgt':
                model_server, model_delta = lgt_aggregation(args, model_server, model_client, model_delta, online_clients_group, online_clients, optimizer, lr, local_steps)
            # elif args.federated_type == 'clgt':
            #     model_server, model_delta = lgt_aggregation(args, model_server, model_client, model_delta, online_clients_group, online_clients, optimizer, lr, local_steps)
            elif args.federated_type == 'scaffold':
                model_server, model_client_control, model_server_control = scaffold_aggregation(args, model_server, model_client, model_server_control, model_client_control,
                                                                                            online_clients_group, online_clients, optimizer, lr, local_steps)
            else:
                model_server = fedavg_aggregation(args, model_server, model_client, online_clients_group, online_clients, optimizer)
             # evaluate the sync time
            logging_sync_time(tracker)

            do_validate(args, model_server, optimizer, criterion, metrics, train_loader, online_clients_group, online_clients, val=False)

            # logging.
            logging_globally(tracker, start_global_time)
            
            # reset start round time.
            start_global_time = time.time()

            # validate the model at the server
            if args.graph.rank == 0:
                do_test(args, model_server, optimizer, criterion, metrics, test_loader)
            log('This round communication time is: {}'.format(args.comm_time[-1]), args.debug)
            # if args.quantized:
            #     log('This round quantization error is: {}'.format(args.quant_error), args.debug)
        else:
            log("Offline in this round. Waiting on others to finish!", args.debug)
        dist.barrier(group=all_clients_group)

    return