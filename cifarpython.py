import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
TESTING = False

def train(net, train_loader, criterion, optimizer, epoch):
    net.train()    
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
    return running_loss

def validate(net, val_loader, criterion):
    correct = 0
    total = 0
    net.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)       
            correct += (predicted== labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            if i % 500 == 0:
                print('Validation Loss: {:.6f}'.format(loss.item()))
    print("Validation Accuracy: ", correct/total)
    return running_loss

def test(net, test_loader):
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)       
            correct += (predicted== labels).sum().item()
            total += labels.size(0)
   
    print("Test Accuracy: ", correct/total)

def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    net = Net()
    if TESTING:
        net.load_state_dict(torch.load("./model"))
        test_dataset = CIFAR10("./", train = False, download = True, transform = transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, 64, True)
        test(net, test_loader)
    
    else:
        
        train_dataset = CIFAR10("./", train = True, download = True, transform = transform)
        valid_dataset = CIFAR10("./", train = True, download = True, transform = transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))

        np.random.seed(77)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        num_train = len(train_idx)
        num_valid = len(valid_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, sampler=valid_sampler)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), 0.01, 0.9)

        train_loss = []
        valid_loss = []
        epochs = np.linspace(0,15,16)
        for epoch in range(16):
            train_loss.append(train(net, train_loader, criterion, optimizer, epoch)/num_train)
            valid_loss.append(validate(net, val_loader, criterion)/num_valid)
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.plot(epochs, valid_loss, label="Validation Loss")
        plt.legend()
        plt.savefig("loss.png")

        torch.save(net.state_dict(), "./model")
   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    main()