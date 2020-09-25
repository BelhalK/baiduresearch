'''Simple example network for MNIST from pytorch docs'''

import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MnistNetSmall(nn.Module):
    def __init__(self):
        super(MnistNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.fc1 = nn.Linear(20 * 12 * 12, 10)
        #self.fc1 = nn.Linear(28 * 28, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 20 * 12 * 12)
        #x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class MnistNetLarge(nn.Module):
    def __init__(self):
        super(MnistNetLarge, self).__init__()
        self.fc1 = nn.Linear(28*28, 900)
        self.fc2 = nn.Linear(900, 900)
        self.fc3 = nn.Linear(900, 900)
        self.fc4 = nn.Linear(900, 500)
        self.fc5 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class MnistNetLarge2(nn.Module):
    def __init__(self):
        super(MnistNetLarge, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 600)
        self.fc2 = nn.Linear(600, 900)
        self.fc3 = nn.Linear(900, 900)
        self.fc4 = nn.Linear(900, 900)
        self.fc5 = nn.Linear(900, 900)
        self.fc6 = nn.Linear(900, 900)
        self.fc7 = nn.Linear(900, 900)
        self.fc8 = nn.Linear(900, 500)
        self.fc9 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        #x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return F.log_softmax(x, dim=1)
