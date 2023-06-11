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
import wandb
import os
import json
from collections import defaultdict
from datasets.femnist import Femnist
import datasets.np_transforms as nptr


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    data['x'] = []
    data['y'] = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            
        for users, d in cdata['user_data'].items():
            for key, values in d.items():
                for value in values:
                  data[key].append(value)
                    
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

def get_train_valid_test_dataloader(train_batch_size, test_batch_size=256):

    # Prepare data transformations and then combine them sequentially

    train_transforms = nptr.Compose([
        nptr.ToTensor(),
        nptr.Normalize((0.5,), (0.5,)),
    ])
    test_transforms = nptr.Compose([
        nptr.ToTensor(),
        nptr.Normalize((0.5,), (0.5,)),
    ])

    # Load data
    niid = False
    train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
    test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')

    train_data , test_data = read_femnist_data(train_data_dir, test_data_dir)

    full_training_data = Femnist(train_data, train_transforms, '')
    test_data = Femnist(test_data, test_transforms, '')

    # Create train and validation splits

    num_samples = len(full_training_data)
    num_training_samples = int(num_samples*0.8+1) # 80% training_data
    num_validation_samples = num_samples - num_training_samples # 20% validation data
    training_data, validation_data = random_split(full_training_data, [num_training_samples,
    num_validation_samples])

    # Initialize dataloaders

    train_dataloader = DataLoader(training_data, train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_data, test_batch_size)
    test_dataloader = DataLoader(test_data, test_batch_size)
    
    return train_dataloader, valid_dataloader, test_dataloader

def main(batch_size=64, device='cuda:0', learning_rate=10**-2, weight_decay=10**-6, momentum=0.9, epochs=50):

    # open a file in which the results are written
    #f = open(f'./results/centralized_setting/results{fileNo}.txt', 'a')
    #f.write(f'Parameters:\n')
    #f.write(f'\t learning_rate: {learning_rate}, weight_decay: {weight_decay}, momentum: {momentum}, epochs: {epochs}\n\n')

    # open wandb and sets all the parameters of this configuration
    
    wandb.init(
        # set the wandb project where this run will be logged
        project = "Centralized_setting",
        name = "bs=" + str(batch_size) + "_" + \
                "lr=" + str(learning_rate) + "_" + \
                "wd=" + str(weight_decay) + "_" + \
                "m=" + str(momentum) + "_" + \
                "e=" + str(epochs) + "_",
        config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "epochs": epochs
        })

    # load the dataset and divedes it in training set, validation set and test set
    train_loader, val_loader, test_loader = get_train_valid_test_dataloader(train_batch_size=batch_size)

    # model
    net = CNN(num_classes=62).to(device)

    # optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    # loss function
    loss_function = get_loss_function()

    for e in range(epochs):

        # training and accuracy on the training set
        train_loss, train_accuracy = train(net, train_loader, optimizer, loss_function, device)

        # test on the validation set
        val_loss, val_accuracy = test(net, val_loader, loss_function, device)

        #f.write(f'Epoch: {e+1}\n')
        #f.write(f'\t Training loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}\n')
        #f.write(f'\t Validation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}\n')
        #f.write('-----------------------------------------------------\n')
        #f.write('After training:\n')

        # test on the training set after training
        train_loss, train_accuracy = test(net, train_loader, loss_function, device)

        # test on the validation set after training
        val_loss, val_accuracy = test(net, val_loader, loss_function, device)

        # test on the test set after training
        test_loss, test_accuracy = test(net, test_loader, loss_function, device)

        #f.write(f'\t Training loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}\n')
        #f.write(f'\t Validation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}\n')
        #f.write(f'\t Test loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}\n\n')
        #f.write('-----------------------------------------------------\n\n')

        wandb.log({"Train Accuracy": train_accuracy, "Train Loss": train_loss,
                   "Validation Accuracy": val_accuracy, "Validation Loss": val_loss,
                   "Test Accuracy": test_accuracy, "Test Loss": test_loss})

    # close the file after all the infos have been written
    # f.close()
    wandb.finish()

if __name__ == '__main__':
       
    # hyperparameters
    param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [10**-2, 10**-1],
        'weight_decay' : [10**-6, 10**-4, 10**-2],
        'momentum' : [0.7, 0.9, 0.99],
        'epochs': [1, 5, 10]
    }

    # creates a dictionary with all the combinations of the parameters
    params = ParameterGrid(param_grid)

    wandb.login()

    # hyperparameters tuning
    for i, param in enumerate(params):
        main(**param)