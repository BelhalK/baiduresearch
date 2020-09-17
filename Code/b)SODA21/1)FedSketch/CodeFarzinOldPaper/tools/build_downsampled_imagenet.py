# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import pickle
from os.path import join

from sklearn.datasets import load_svmlight_file

import numpy as np
from tensorpack.dataflow import dataset, PrefetchDataZMQ, LMDBSerializer


def get_args():
    parser = argparse.ArgumentParser(description='aug data.')

    # define arguments.
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--data_type', default='train', type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    return args


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def sequential_downsampled_imagenet(root_path, data_type):
    data = DownsampledImageNet(root_path, data_type)
    lmdb_file_path = join(root_path, '{}.lmdb'.format(data_type))

    ds1 = PrefetchDataZMQ(data, nr_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


class DownsampledImageNet(object):
    def __init__(self, root_path, data_type):
        # define data path.
        tr_lmdb_path = join(root_path, 'train.lmdb')
        val_lmdb_path = join(root_path, 'val.lmdb')

        self.img_size = 32
        self.img_size_square = self.img_size * self.img_size

        # get dataset.
        list_of_tr_data = [
            unpickle(
                join(root_path, 'raw_data', 'train_data_batch_{}'.format(idx)))
            for idx in range(1, 11)
        ]
        te_data = unpickle(join(root_path, 'raw_data', 'val_data'))

        # extract features.
        self.features, self.labels = self._get_images_and_labels(
            list_of_tr_data, te_data, data_type)

    def _get_images_and_labels(self, list_of_tr_data, te_data, data_type):
        def _helper(_feature, _target, _mean):
            # process data.
            _feature = _feature - _mean
            _target = [x - 1 for x in _target]
            return _feature, _target

        # deal with train data.
        features, labels = [], []
        if 'train' in data_type:
            print('process train data.')

            for tr_data in list_of_tr_data:
                # extract raw data.
                _feature = tr_data['data']
                _target = tr_data['labels']
                _mean = tr_data['mean']

                # get data.
                feature, target = _helper(_feature, _target, _mean)

                # store data.
                features.append(feature)
                labels.append(target)
        elif 'val' in data_type:
            print('process val data.')

            _feature = te_data['data']
            _target = te_data['labels']
            _mean = list_of_tr_data[0]['mean']

            # get data.
            feature, target = _helper(_feature, _target, _mean)

            # store data.
            features.append(feature)
            labels.append(target)

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        # deal with val data.
        return features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            if self.features[k] is not None and self.labels[k] is not None:
                feature = cv2.imencode('.jpeg', self.features[k])[1]
                yield [feature, self.labels[k]]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        pass


def main(args):
    sequential_downsampled_imagenet(args.data_dir, args.data_type)


if __name__ == '__main__':
    args = get_args()
    main(args)
