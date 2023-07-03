from numpy.core.fromnumeric import nonzero
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import ParameterGrid
from models.cnn import CNN
from main import get_datasets
from utils.args import get_parser
from torch.utils.data import ConcatDataset
import wandb
import os
import json
from collections import defaultdict
from datasets.femnist import Femnist

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
    return data

def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)


def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function

def get_optimizer(net, lr, wd, momentum):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum) # Stochastic Gradient Descent algorithm
    return optimizer

def train(net, dataloader, optimizer, loss_function, device='cuda:0'):
    samples = 0
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.train() # Strictly needed if network contains layers which has different behaviours between train and test

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device) # Load data into GPU
        outputs = net(inputs) # Forward pass
        loss = loss_function(outputs,targets) # Apply the loss
        loss.backward() # Backward pass
        optimizer.step() # Update parameters
        optimizer.zero_grad() # Reset the optimizer
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        samples += inputs.shape[0]
        cumulative_accuracy += predicted.eq(targets).sum().item()
    
    return cumulative_loss/samples, cumulative_accuracy/samples*100

def test(net, dataloader, loss_function, device='cuda:0'):
    samples = 0
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.eval() # Strictly needed if network contains layers which have different behaviours between train and test

    with torch.no_grad(): # Disables gradient computation
        for batch_idx, (inputs, targets) in enumerate(dataloader):
        
            inputs, targets = inputs.to(device), targets.to(device) # Load data into GPU
            outputs = net(inputs) # Forward pass
            loss = loss_function(outputs,targets) # Apply the loss
            samples += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
    return cumulative_loss/samples, cumulative_accuracy/samples*100

def get_train_test_dataloader(domain_out, train_batch_size, test_batch_size=256):

    # Load data
    niid = False
    train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
    test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')

    train_data , test_data = read_femnist_data(train_data_dir, test_data_dir)
    
    train_datasets, test_datasets = [], []
    
    rotation = [0, 15, 30, 45, 60, 75]
    n_clients_per_set = int(round(1002 / 6, 0))
    n_clients_total = 1002
    print(f"Clients per set: {n_clients_per_set}")
    cont_clients = 0
    for user, data in train_data.items():
        if cont_clients >= n_clients_total:
            if domain_out != None:
                break
            else:
                train_transforms = nptr.Compose([
                    nptr.ToTensor(),
                    nptr.Normalize((0.5,), (0.5,)),
                ])
                train_datasets.append(Femnist(data, train_transforms, user))
                continue
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            sstr.RandomRotation(rotation[int(cont_clients / n_clients_per_set)]),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        cont_clients += 1
        train_datasets.append(Femnist(data, train_transforms, user))

    if domain_out == None:
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,))
        ])
        for user, data in test_data.items():
            test_datasets.append(Femnist(data, test_transforms, user))
    else:
        test_datasets = train_datasets[int(domain_out*len(train_datasets)/6) : int((domain_out+1)*len(train_datasets)/6)]
        del train_datasets[int(domain_out*len(train_datasets)/6) : int((domain_out+1)*len(train_datasets)/6)]
    
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, test_batch_size)
    
    return train_dataloader, test_dataloader



def main(domain_out = None, batch_size=64, device='cuda:0', learning_rate=10**-2, weight_decay=10**-6, momentum=0.9, epochs=50):
    
    if domain_out != None:
        name = "domain_out = " + str(domain_out*15) + "°",
    else:
        name = "domain_out = None"

    wandb.init(
        project = "Rotated_centralized",
        name = name,
        config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "epochs": epochs
        })

    # load the dataset and divedes it in training set and test set
    train_loader, test_loader = get_train_test_dataloader(domain_out, train_batch_size=batch_size)

    # model
    net = CNN(num_classes=62).to(device)

    # optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    # loss function
    loss_function = get_loss_function()

    for e in range(epochs):
        
        print(f"Epoch: {e}")

        # training and accuracy on the training set
        print("Training")
        train_loss, train_accuracy = train(net, train_loader, optimizer, loss_function, device)

        # test on the test set after training
        print("Testing")
        test_loss, test_accuracy = test(net, test_loader, loss_function, device)

        wandb.log({"Train Accuracy": train_accuracy, "Train Loss": train_loss,
                   "Test Accuracy": test_accuracy, "Test Loss": test_loss})

    wandb.finish()

if __name__ == '__main__':
       
    domain_target_set = [0, 1, 2, 3, 4, 5, None] # 0°, 15°, 30°, 45°, 60°, 75°, None (to run the centralized on all 
                                                 # the train set in which only 1000 clients are rotated and using as test the testset)
    wandb.login()

    # 6 runs in which one of the 6 rotation domain is left out and used as test set
    for i, domain_out in enumerate(domain_target_set):
        main(domain_out)