# python -m eval --path path/to/dataset1 --path path/to/dataset2 --gpu

from utils.fid import calculate_fid_given_paths
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))


print("==> Parse Arguments...")

args = parser.parse_args()

#### Evaluation Metrics
path = args.path #path to the generated images or npz file (put to path for FD comparison)
batch_size = args.batch_size #bs to use
gpu = args.gpu
dims = args.dims #Dimensionality of Inception features to use.

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

print("==> Start FID Evaluation...")
fid_value = calculate_fid_given_paths(path,batch_size,gpu != '',dims)
print('FID: ', fid_value)


print("==> Task is Finished...")