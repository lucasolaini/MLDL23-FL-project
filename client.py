import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

from utils.utils import HardNegativeMining, MeanReduction
import torch.nn.functional as F
import torch.distributions as distributions


class Client:

    def __init__(self, args, dataset, model, p, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        
        if args.FedVC:
            NVC = 192
            if len(dataset) >= NVC:
                train_idxs_vc = torch.randperm(len(self.dataset))[:NVC]
            else:
                train_idxs_vc = torch.randint(len(self.dataset), (NVC, ))
                
            self.train_loader = DataLoader(Subset(self.dataset, train_idxs_vc), batch_size=self.args.bs, shuffle=True, drop_last=True) \
                if not test_client else None
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
                if not test_client else None
                
        self.test_loader = DataLoader(self.dataset, batch_size=1024, shuffle=False)
        
        if args.FedSR:
            self.net = model
            self.net.fc1 = nn.Sequential()
            self.cls = nn.Linear(1024, 62)
            self.model = nn.Sequential(self.net, self.cls)
            self.net.cuda()
            self.cls.cuda()
        else:
            self.model = model
            
        if args.FedIR:
            labels = set(torch.arange(62))
            self.q = torch.tensor([(torch.tensor(self.dataset.targets) == label).sum() for label in labels]) / len(self.dataset)
            self.p = p
            
            weight = torch.zeros(62).cuda()
            
            for i in range(62):
                if self.q[i] != 0: # some labels might not be seen in this dataset
                    weight[i] = self.p[i] / self.q[i]
                else:
                    weight[i] = 0.001 # setting weight = 0 leads to overfitting
        else:
            weight = None
            
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=weight)
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()    
            
        

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        if self.args.model == 'cnn':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(), labels.cuda() # Load data into GPU
            outputs = self._get_outputs(images) # Forward pass
            loss = self.criterion(outputs, labels) # Apply the loss
            loss.backward() # Backward pass
            optimizer.step() # Update parameters
            optimizer.zero_grad() # Reset the optimizer

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)

        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer)

        return len(self.dataset), copy.deepcopy(self.model.state_dict())

        

    def test(self, metric, set):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()
        
        if set == 'eval':
            dataloader = self.train_loader
        elif set == 'test':
            dataloader = self.test_loader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.cuda(), labels.cuda()
                if self.args.FedSR:
                  z = self.featurize(images)
                  outputs = self.cls(z)
                else:
                  outputs = self._get_outputs(images)
                self.update_metric(metric, outputs, labels)
                
    def compute_loss(self):
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.model.eval()
        loss = 0
        cnt = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs = self._get_outputs(images)
                loss += loss_fn(outputs, labels).item()
                cnt += labels.size(0)
        
        return loss
        
    def featurize(self, x, num_samples=1, return_dist=False):
        z_params = self.net(x) # 64 x 2048
        z_mu = z_params[:,:1024]
        z_sigma = F.softplus(z_params[:,1024:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample([num_samples]).view([-1, 1024])
        
        if return_dist:
            return z, (z_mu,z_sigma)
        else:
            return z
        
    def run_epoch_FedSR(self, cur_epoch, optimizer):
        
        r_mu = nn.Parameter(torch.zeros(62, 1024)).cuda()
        r_sigma = nn.Parameter(torch.ones(62, 1024)).cuda()
        C = nn.Parameter(torch.ones([])).cuda()
        
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(), labels.cuda() 
            z, (z_mu,z_sigma) = self.featurize(images, return_dist=True)
            outputs = self.cls(z)
            loss = self.criterion(outputs, labels)
           
            obj = loss
            regL2R = torch.zeros_like(obj)
            regCMI = torch.zeros_like(obj)
            if self.args.L2R_coeff != 0.0:
                regL2R = z.norm(dim=1).mean()
                obj = obj + self.args.L2R_coeff*regL2R
            if self.args.CMI_coeff != 0.0:
                r_sigma_softplus = F.softplus(r_sigma)
                r_mu = r_mu[labels]
                r_sigma = r_sigma_softplus[labels]
                z_mu_scaled = z_mu*C
                z_sigma_scaled = z_sigma*C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                        (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                regCMI = regCMI.sum(1).mean()
                obj = obj + self.args.CMI_coeff*regCMI
                
            optimizer.zero_grad()
            obj.backward(retain_graph=True)
            optimizer.step()
        
    def train_FedSR(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)

        for epoch in range(self.args.num_epochs):
            self.run_epoch_FedSR(epoch, optimizer)

        return len(self.dataset), copy.deepcopy(self.model.state_dict())