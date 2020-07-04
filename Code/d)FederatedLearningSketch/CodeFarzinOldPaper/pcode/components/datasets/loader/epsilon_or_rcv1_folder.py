# -*- coding: utf-8 -*-

from pcode.tracking.logging import log
from pcode.components.datasets.loader.utils import IMDBPT


def define_epsilon_or_rcv1_or_MSD_folder(args,root):
    log('load epsilon_or_rcv1_or_MSD from lmdb: {}.'.format(root), args.debug)
    return IMDBPT(root, is_image=False)
