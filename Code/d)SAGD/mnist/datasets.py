from torchvision import datasets
from torchvision.datasets.utils import *
from torchvision.datasets.mnist import read_image_file, read_label_file
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import gzip
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class MNIST(datasets.VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        trans_targets = []
        for target in targets:
            target = int(target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            trans_targets.append(target)

        trans_data = []
        for img in data:
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            trans_data.append(img)
        self.data = trans_data
        self.targets = trans_targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        """
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        """

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # os.makedirs(self.raw_folder)
        # os.makedirs(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class SubData(object):
    def __init__(self, data, targets):

        label_data = [[] for i in range(10)]
        for cur_data, cur_target in zip(data, targets):
            label_data[cur_target].append(cur_data)

        self.label_data = label_data
        self.data = []
        self.targets = []
    
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
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)



class SubMNIST(MNIST):
    def __init__(
            self, root, train=True, transform=None,
            target_transform=None, download=False,
            sample_num=0):
        super(SubMNIST, self).__init__(root, train, transform, target_transform, download)
        if (sample_num == 0):
            return
        # the sample num in each label class
        label_sample_num = sample_num // 10

        # put the data into different labels
        label_data = [[] for i in range(10)]

    
        for cur_data, cur_target in zip(self.data, self.targets):
            label_data[cur_target].append(cur_data)

        label_sample_data = [[] for i in range(10)]
        for idx in range(10):
            label_data_num = len(label_data[idx])
            label_idxs = [i for i in range(label_data_num)]
            np.random.shuffle(label_idxs)

            for j in label_idxs[:label_sample_num]:
                label_sample_data[idx].append(label_data[idx][j])

        sample_data = []
        sample_target = []

        for idx in range(10):
            for cur_sample in label_sample_data[idx]:
                sample_data.append(cur_sample)
                sample_target.append(idx)
        self.data = sample_data
        self.targets = sample_target

        print('subdataset size', len(self.data))
