# a logger

from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import os
print(os.getcwd())

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name], dtype='float'))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None, scaling=None):
        names = self.names if names == None else names
        if scaling == None:
            def scaling(x): return 1
            ylab = ""
        elif scaling == "normalized":
            def scaling(x): return np.linalg.norm(x)
            ylab = "normalized"
        else:
            scaling

        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            y = np.asarray(numbers[name], dtype='float')
            plt.plot(x, y / scaling(y))
        #plt.legend([self.title + '(' + name + ')' for name in names])
        plt.legend([name for name in names])
        plt.grid(True)
        plt.title(self.title)
        plt.xlabel("epoch")
        plt.ylabel(ylab)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        #plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        #plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(legend_text)
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
                    
if __name__ == '__main__':
    # Test Logger
        logger = Logger('checkpoint/test/test.txt')
        logger.set_names(['Train loss', 'Valid loss','Test loss'])
        
        length = 100
        t = np.arange(length)
        train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
        valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
        test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
        
        for i in range(0, length):
            logger.append([train_loss[i], valid_loss[i], test_loss[i]])
        logger.plot()
        savefig('checkpoint/test/test.eps')

    # Example: logger monitor
        # paths = {
        #     'ResNet18': 'checkpoint/checkpoint_cifar10/logResNet18_v2.txt',
        #     'ShuffleNetV2': 'checkpoint/checkpoint_cifar10/logShuffleNetV2_v2.txt',
        #     'ResNeXt29':'checkpoint/checkpoint_cifar10/logResNeXt29.txt'
        # }

        # paths = {
        #     'ResNet18': 'checkpoint/checkpoint_cifar10/logcifar.resnet.sign5_mom9.momSwitch.072319.txt'
        # }
        # field = ['Valid Loss']
        # monitor = LoggerMonitor(paths)
        # monitor.plot(names=field)
        # savefig('checkpoint/checkpoint_cifar10/resnet.eps')