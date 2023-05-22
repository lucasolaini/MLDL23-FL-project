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
            updates.append(parameters())
            return updates
        
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        return np.sum(updates, axis=0)


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        print(f'Train clients: {len(self.train_clients)}')
        print(f'Test clients: {len(self.test_clients)}')
        for r in range(self.args.num_rounds):
            print(f'Round {r}')
            clients_selected = self.select_clients()
            updates = self.train_round(clients_selected)
            print(len(updates))
            aggregated_updates = self.aggregate(updates)
            self.model.load_state_dict(OrderedDict(aggregated_updates))
            self.eval_train()
            self.test()
            self.metrics.get_results()
            self._show_result()


    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        for i, c in enumerate(self.train_clients):
            c.test(self.metrics)

    def test(self):
        """
        This method handles the test on the test clients
        """
        for i, c in enumerate(self.test_clients):
            c.test(self.metrics)

    def _show_results(self):
        print(self.metrics.results)
