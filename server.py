import copy
from collections import OrderedDict

import numpy as np
import torch
import wandb


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        
        self.wandb_run_id = ''

    def select_clients(self, strategy='uniform'):
        if strategy == 'uniform':
            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            return np.random.choice(self.train_clients, num_clients, replace=False)
        if strategy == 'high':
            prob = 0.5
            frac = 0.1
            clients_fraction = round(self.train_clients.size * frac)
            remaining_clients = self.train_clients.size - clients_fraction
            p = [prob / clients_fraction] * clients_fraction + [(1 - prob) / remaining_clients] * remaining_clients
            return np.random.choice(self.train_clients, num_clients, replace=False, p=p)
        if strategy == 'low':
            prob = 0.0001
            frac = 0.3
            clients_fraction = round(self.train_clients.size * frac)
            remaining_clients = self.train_clients.size - clients_fraction
            p = [prob / clients_fraction] * clients_fraction + [(1 - prob) / remaining_clients] * remaining_clients
            return np.random.choice(self.train_clients, num_clients, replace=False, p=p)
        else:
            raise NotImplementedError


    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        dataset_lengths = []

        for i, c in enumerate(clients):
            # update training client's model
            c.model.load_state_dict(self.model.state_dict())
            
            dataset_length, parameters = c.train()
            updates.append(parameters)
            dataset_lengths.append(dataset_length)

        total_dataset_lengths = sum(dataset_lengths)

        for i, update in enumerate(updates): 
            for k, v in update.items():
                update[k] = update[k] * (dataset_lengths[i] / total_dataset_lengths)

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
        
        self.setup_wandb()

        for r in range(self.args.num_rounds):
            print(f'Round {r}')
            clients_selected = self.select_clients()
            updates = self.train_round(clients_selected)

            aggregated_state_dict = self.aggregate(updates)

            self.model.load_state_dict(aggregated_state_dict)
            torch.save(self.model.state_dict(), self.args.backup_folder + '/' + self.wandb_run_id)

            # self.eval_train()
            self.test()

            wandb.log({"Overall Accuracy": self.metrics['test'].results['Overall Acc'] * 100, \
                       "Mean Accuracy": self.metrics['test'].results['Mean Acc'] * 100})
            
        wandb.finish()

    def update_client_state(self, update):
        for c in self.train_clients:
            c.model.load_state_dict(update)

        for c in self.test_clients:
            c.model.load_state_dict(update)

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        for i, c in enumerate(self.train_clients):
            print(f'Training client {i}: {c.name}')
            c.test(self.metrics['eval_train'])

        print(f'Evaluation on train clients')
        self.metrics['eval_train'].get_results()
        print(self.metrics['eval_train'].__str__())

    def test(self):
        """
        This method handles the test on the test clients
        """
        for i, c in enumerate(self.test_clients):

            # update test client's model
            c.model.load_state_dict(self.model.state_dict())
            c.test(self.metrics['test'])

        self.metrics['test'].get_results()
        
        
    def setup_wandb(self):
        """
        This method sets up wandb
        """
        
        wandb.login()
        
        # select the correct project in wandb
        if self.args.niid:
            project = "Federated_setting_niid"
        else:
            project = "Federated_setting_iid"
            
        project = "prova"
        
        # assings a name to the run    
        name = "bs=" + str(self.args.bs) + "_" + \
               "lr=" + str(self.args.lr) + "_" + \
               "wd=" + str(self.args.wd) + "_" + \
               "m=" + str(self.args.m) + "_" + \
               "e=" + str(self.args.num_epochs) +"_" + \
               "cpr=" + str(self.args.clients_per_round)
        
        # select the current configuration       
        config = {
            "niid": self.args.niid,
            "learning_rate": self.args.lr,
            "weight_decay": self.args.wd,
            "momentum": self.args.m,
            "batch_size": self.args.bs,
            "rounds": self.args.num_rounds,
            "epochs": self.args.num_epochs,
            "clients_per_round": self.args.clients_per_round,
            "model": self.model._get_name()
        }
         
        if self.args.backup:
            wandb.init(project=project, id=self.args.run_id, resume="must")
            if wandb.run.resumed:
                print(f"Resuming run {wandb.run.id}")
                self.model.load_state_dict(torch.load(self.args.backup_path))
        else:
            wandb.init(project=project, name=name, config=config)
        
        self.wandb_run_id = wandb.run.id