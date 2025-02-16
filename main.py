import warnings
import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from models.cnn import CNN


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        model = CNN(num_classes=get_dataset_num_classes(args.dataset))
        return model


def get_transforms(args, rotated = 0):  
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            sstr.RandomRotation(rotated),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            #sstr.RandomRotation(rotated),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


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


def get_datasets(args):
    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)
        
        train_datasets, test_datasets = [], []

        if args.dom_gen:
            rotation = [0, 15, 30, 45, 60, 75]
            n_clients_per_set = int(round(1002 / 6, 0))
            # select 1002 clients (divisible by 6 -> groups have the same dimension)
            n_clients_total = 1002
            cont_clients = 0

            
            for user, data in train_data.items():
                if cont_clients >= n_clients_total:
                    if args.leave_one_out != None: # train set only for 1002 clients
                        break
                    else:
                        train_transforms, test_transforms = get_transforms(args, rotation[0]) 
                        train_datasets.append(Femnist(data, train_transforms, user))
                        continue
                train_transforms, test_transforms = get_transforms(args, rotation[int(cont_clients / n_clients_per_set)])
                cont_clients += 1
                train_datasets.append(Femnist(data, train_transforms, user)) 
            # same test set as FedAvg  
            for user, data in test_data.items():
                test_datasets.append(Femnist(data, test_transforms, user))
        else:
            train_transforms, test_transforms = get_transforms(args)
            for user, data in train_data.items():
                train_datasets.append(Femnist(data, train_transforms, user))
            for user, data in test_data.items():
                test_datasets.append(Femnist(data, test_transforms, user))

    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model, p):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, p, test_client=i == 1))
    return clients[0], clients[1]

def get_label_distribution(train_datasets):
    labels = set(torch.arange(62))
    p = torch.zeros(62)
    tot_len = 0
    
    for dataset in train_datasets:
        p += torch.tensor([(torch.tensor(dataset.targets) == label).sum() for label in labels])
        tot_len += len(dataset)
        
    return p / tot_len

def main():
    parser = get_parser()
    args = parser.parse_args() # parses the arguments recived from command line
    set_seed(args.seed) # sets the seed to the one specified as argument

    print(f'Initializing model...')
    model = model_init(args)
    if torch.cuda.is_available():
        model.cuda()
    else:
         warnings.warn("Gpu not available: moving on cpu")
         model.cpu()   
    print('Done.')

    print('Generate datasets...')
    train_datasets, test_datasets = get_datasets(args)
    print('Done.')   
    
    if args.FedIR:
        p = get_label_distribution(train_datasets)
    else:
        p = None

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model, p)

    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()


if __name__ == '__main__':
    main()
