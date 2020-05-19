from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from torchvision import datasets
from torchvision.datasets.utils import check_integrity


class SubData(object):
    def __init__(self, data, targets, transform=None, target_transform=None):
        label_data = [[] for i in range(10)]
        for cur_data, cur_target in zip(data, targets):
            label_data[cur_target].append(cur_data)

        self.label_data = label_data

        for lidx, data in enumerate(label_data):
            print('class {} has {} samples'.format(lidx, len(data)))
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
    
    def set_sub_sample(self, sample_num):
        label_sample_num = sample_num // 10
        label_sample_data = [[] for i in range(10)]
        for idx in range(10):
            label_data_num = len(self.label_data[idx])
            label_idxs = [i for i in range(label_data_num)]
            np.random.shuffle(label_idxs)
            for j in label_idxs[:label_sample_num]:
                label_sample_data[idx].append(self.label_data[idx][j])

        sample_data = []
        sample_target = []

        for idx in range(10):
            for cur_sample in label_sample_data[idx]:
                sample_data.append(cur_sample)
                sample_target.append(idx)

        data_idxs = [i for i in range(len(sample_data))]
        np.random.shuffle(data_idxs)

        self.data = [sample_data[idx] for idx in data_idxs]
        self.targets = [sample_target[idx] for idx in data_idxs]
    
        # print(len(self.data))
        # import pdb; pdb.set_trace()
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)