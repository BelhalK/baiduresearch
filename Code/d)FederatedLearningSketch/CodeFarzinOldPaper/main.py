# -*- coding: utf-8 -*-
import pcode.components.datasets.tensorpack.serialize as serialize
import platform

import torch.distributed as dist
import torch.multiprocessing as mp
import os

from parameters import get_args
from pcode.components.create_components import create_components
from pcode.utils.init_config import init_config
from pcode.flow.federated_running import train_and_validate_federated_complete
from pcode.tracking.logging import log, configure_log, log_args




def init_process(rank, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29506'
    dist.init_process_group(args.dist_backend, rank=rank, world_size=args.num_workers)
    main(args)
    return

def main(args):
    """distributed training via mpi backend."""
    # dist.init_process_group('mpi')

    # init the config.
    init_config(args)
    
    print("Config is initialized")
    # create model and deploy the model.
    model, criterion, scheduler, optimizer, metrics = create_components(args)
    # config and report.
    configure_log(args)
    log_args(args, args.debug)
    log(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.ranks_with_blocks[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_cuda else 'CPU',
            args.graph.device
            )
        , args.debug)
    # train and evaluate model.
    if args.federated:       
        train_and_validate_federated_complete(args, model, criterion, scheduler, optimizer, metrics)
    else:
        raise NotImplementedError

    return 

if __name__ == '__main__':
    args = get_args()
    # mp.spawn(init_process, nprocs=args.num_workers, args=(args,))
    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=init_process, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
