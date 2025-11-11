# imports
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from src.resnet import ResNet18
import wandb
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context # get around security warning

#wandb logging
wandb.init(project="resnet-cifar10-sprint", name="resnt18-scratch")

#data transforms
transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomCrop(32, padding=4), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                     (0.2023, 0.1994, 0.2010)),])
# load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=30)

#train one epoch
def train_epoch(): 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0 
    for i , (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if random.random() < 0.5:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size(0))
            inputs_mix = lam * inputs + (1 - lam) * inputs[rand_index]
            targets_a, targets_b = targets, targets[rand_index]
            outputs = model(inputs_mix)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % 100 == 99:
            acc = 100. * correct / total
            print(f"[{i+1:3d}] loss: {running_loss/100: .3} | acc: {acc: .2f}%")
            wandb.log({"train_loss": running_loss/100, "train_acc": acc})
            running_loss = 0.0
            correct = 0
            total = 0

# test accuracy
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc: .2f}%")
    wandb.log({"test_acc": acc})
    return acc

# run 10 epochs
if __name__== "__main__":
    print("Starting Training...")
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        train_epoch()
        test()
        scheduler.step()    
    wandb.finish()
    print("Training Complete.")

