# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

from pcode.tracking.checkpoint import init_checkpoint
from pcode.flow.communication import configure_sync_scheme
from pcode.utils.topology import FCGraph


def set_local_stat(args):
    args.local_index = 0
    args.client_epoch_total = 0
    args.block_index = 0
    args.global_index = 0
    args.local_data_seen = 0
    args.best_prec1 = 0
    args.best_epoch = []
    args.rounds_comm = 0
    args.tracking = {'cosine': [], 'distance': []}
    args.comm_time = []

def init_config(args):
    # define the graph for the computation.
    cur_rank = dist.get_rank()
    args.graph = FCGraph(cur_rank, args.blocks, args.on_cuda, args.world)

    if args.graph.rank != 0:
        args.debug=False


    if args.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.set_device(args.graph.device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # local conf.
    set_local_stat(args)

    # define checkpoint for logging.
    init_checkpoint(args)

    # define sync scheme.
    configure_sync_scheme(args)
