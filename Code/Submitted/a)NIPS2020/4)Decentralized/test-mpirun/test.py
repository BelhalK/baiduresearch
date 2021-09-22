#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name() # get the name of the node
print("World Size is {}.".format(size))
print("Hello world from process {} of {} at {}.".format(rank, size, node_name))
if rank == 0:
    data = range(10)
    print("process {} bcast data {} to other processes".format(rank, data))
else:
    data = None
data = comm.bcast(data, root=0)
print("process {} recv data {}...".format(rank, data))
data = [i + rank for i in data]
print("process {} prepare data {}...".format(rank, data))
new_data = comm.gather(data, root=0)
print("after gather, process {} new_data {}...".format(rank, new_data))