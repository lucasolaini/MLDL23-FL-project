import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []

        for i, c in enumerate(clients):
           
            dataset_length, parameters = c.train()
            updates.append(parameters)
        
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        aggregated_update = None

        for update in updates:
            if aggregated_update == None:
                aggregated_update = update
            else:
                for k, v in update.items():
                    aggregated_update[k] += v

        return aggregated_update

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        print(f'Number of train clients: {len(self.train_clients)}')
        print(f'Number of test clients: {len(self.test_clients)}')
        for r in range(self.args.num_rounds):
            print(f'Round {r}')
            clients_selected = self.select_clients()
            updates = self.train_round(clients_selected)
            aggregated_state_dict = self.aggregate(updates)
            self.model_params_dict = aggregated_state_dict
            self.update_client_state(clients_selected, aggregated_state_dict)
            self.model.load_state_dict(aggregated_state_dict)
            self.eval_train()
            self.test()

    def update_client_state(self, clients, update):
        for c in clients:
            c.model.load_state_dict(update)

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        for i, c in enumerate(self.train_clients):
            c.test(self.metrics['eval_train'])

        print(f'Train set')
        self.metrics['eval_train'].get_results()
        print(self.metrics['eval_train'].__str__())

    def test(self):
        """
        This method handles the test on the test clients
        """
        for i, c in enumerate(self.test_clients):
            c.test(self.metrics['test'])

        print(f'Test set')
        self.metrics['test'].get_results()
        print(self.metrics['test'].__str__())
