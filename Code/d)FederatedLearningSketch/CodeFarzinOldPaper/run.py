import pcode.components.datasets.tensorpack.serialize as serialize
import os
import argparse

def main(args):
    MODEL={'mnist':'mlp'}
    
    NUM_NODES=2
    NUM_WORKER_PER_NODE =  int(args.num_clients / NUM_NODES)
    NUM_WORKERS_NODE = [NUM_WORKER_PER_NODE] * NUM_NODES 

    BLOCKS=(',').join([str(i) for i in NUM_WORKERS_NODE])
    WORLD = ",".join([ ",".join([str(x) for x in range(i)]) for i in NUM_WORKERS_NODE])


    script_params = {
        '--arch':  MODEL[args.dataset],
        '--mlp_num_layers': 2,
        '--mlp_hidden_size': 250,
        '--avg_model': True,
        '--experiment': 'demo',
        '--debug': True,
        '--data': args.dataset,
        '--pin_memory': True,
        '--federated': True,
        '--federated_type':'fedavg',
        '--num_comms':args.num_comms,
        '--federated_sync_type':args.federated_sync_type,
        '--online_client_rate':args.online_client_rate,
        '--num_epochs_per_comm': args.num_epochs_per_comm,
        '--num_class_per_client':args.num_class_per_client,
        '--iid_data':args.iid,
        '--quantized':args.quantized,
        '--unbalanced':args.unbalanced,
        '--batch_size':args.batch_size,
        '--eval_freq': 1, 
        '--partition_data': True,
        '--reshuffle_per_epoch': False,
        '--stop_criteria': "epoch",
        '--num_epochs': args.num_epochs_per_comm * args.num_comms,
        '--on_cuda': False,
        '--num_workers': args.num_clients, 
        '--blocks': BLOCKS,
        '--world': WORLD,
        '--weight_decay': args.weight_decay,
        '--use_nesterov': False,
        '--in_momentum': False,
        '--in_momentum_factor': 0.9, 
        '--out_momentum': False, 
        '--out_momentum_factor': 0.9,
        '--local_step': args.local_steps,
        '--data_dir': args.data_path,
        '--checkpoint': args.data_path
    }
       
    learning_rate = {
        '--lr_schedule_scheme': 'custom_multistep',
        '--lr_change_epochs': ','.join([str(x) for x in range(1,100)]),
        '--lr_warmup': False,
        '--lr': args.lr_gamma ,
        '--lr_warmup_epochs': 3,
        '--lr_decay':1.01,
    }

    # learning_rate = {
    #     '--lr_schedule_scheme':'custom_convex_decay',
    #     '--lr_gamma': args.lr_gamma,
    #     '--lr_mu': args.lr_mu,
    #     '--lr_alpha': 1,
    # }

    script_params.update(learning_rate)


    cmd = 'python main.py '
    for k, v in script_params.items():
        if v is not None:
            cmd += ' {} {} '.format(k, v)

    # run the cmd.
    print('\nRun the following cmd:\n' + cmd)
    os.system(cmd)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
        description='Running distributed optimization in CPU')
  
  parser.add_argument('-e', '--num_epochs_per_comm', default=1, type=int)
  parser.add_argument('-n', '--num_clients', default=20, type=int)
  parser.add_argument('-d', '--dataset', default='fashion_mnist', type=str)
  parser.add_argument('-y', '--lr_gamma', default=1.0, type=float)
  parser.add_argument('-m', '--lr_mu', default=1, type=float)
  parser.add_argument('-b', '--batch_size', default=50, type=int)
  parser.add_argument('-c', '--num_comms', default=100, type=int)
  parser.add_argument('-k', '--online_client_rate', default=1.0, type=float)
  parser.add_argument('-p', '--data_path', default='./data', type=str)
  parser.add_argument('-r', '--num_class_per_client', default=1, type=int)
  parser.add_argument('-t', '--federated_type', default='fedavg', type=str)
  parser.add_argument('-i', '--iid', action='store_true')
  parser.add_argument('-s', '--local_steps', default=1, type=int)
  parser.add_argument('-f', '--federated_sync_type', default='epoch', type=str, choices=['epoch', 'local_step'])
  parser.add_argument('-w', '--weight_decay', default=1e-4, type=float)
  parser.add_argument('-u', '--unbalanced', action='store_true')
  parser.add_argument('-q', '--quantized', action='store_true')

  args = parser.parse_args()

  main(args)