# Importing Required Libraries
import torch
import numpy as np

class TrainTest():
    """
    Class to define the Training and Testing Utility Functions

    Methods
    -------
    get_criterion(criterion_id)
        Returns Criterion

    get_optimizer(optimizer_id, optimizer_config)
        Returns Optimizer

    get_scheduler(scheduler_id, scheduler_config)
        Returns Scheduler
    """
    def __init__(self, data, model, train_config, device, verbose = True):
        self.train_loader, self.val_loader, self.test_loader = data
        self.model = model.to(device)
        self.verbose = verbose
        self.num_epochs = train_config['num_epochs']
        self.criterion = self.get_criterion(train_config['criterion'])
        self.optimizer = self.get_optimizer(train_config['use_optimizer'], 
                                            train_config['optimizers'][train_config['use_optimizer']])
        self.scheduler = self.get_scheduler(train_config['use_scheduler'], 
                                            train_config['schedulers'][train_config['use_scheduler']])
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.metrics_dict = {'train_loss': None, 'train_accuracy': None, 
                             'val_loss': None, 'val_accuracy': None,
                             'test_loss': None, 'test_accuracy': None}
        self.file_name = train_config['model_store']['session_model']

    def get_criterion(self, criterion_id):
        if criterion_id == 'cross_entropy_loss':
            return(torch.nn.CrossEntropyLoss())

    def get_optimizer(self, optimizer_id, optimizer_config):
        if optimizer_id == 'sgd':
            return(torch.optim.SGD(self.model.parameters(), lr=optimizer_config['lr'], 
                                   momentum=optimizer_config['momentum'], 
                                   weight_decay=optimizer_config['weight_decay'], 
                                   nesterov=optimizer_config['nesterov']))

    def get_scheduler(self, scheduler_id, scheduler_config):
        if scheduler_id == 'reduce_lr_on_plateau':
            return(torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=scheduler_config['mode'], 
                                                              factor=scheduler_config['factor'], 
                                                              patience=scheduler_config['patience'], verbose=self.verbose))
        elif scheduler_id == 'zigzag_lr_on_plateau':
            return(ZigZagLROnPlateau(mode=scheduler_config['mode'], 
                                     up_factor=scheduler_config['up_factor'], down_factor=scheduler_config['down_factor'],
                                     up_patience=scheduler_config['up_patience'], down_patience=scheduler_config['down_patience']))

    def train(self):
        print(f"\nTraining for {self.num_epochs} epochs:\n")
        for epoch in range(self.num_epochs):
            train_accuracy, train_loss = self.train_step(self.train_loader)
            val_accuracy, val_loss = self.eval_step(self.val_loader)
            metric = self.__get_scheduler_metric(train_accuracy, train_loss, val_accuracy, val_loss)
            self.scheduler.step(metric)
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)
            if self.verbose:
                print(f"\n\tEpoch: {epoch+1}/{self.num_epochs}")
                print(f"\tTraining Loss: {round(train_loss, 4)}; Training Accuracy: {round(train_accuracy*100, 4)}%")
                print(f"\tValidation Loss: {round(val_loss, 4)}; Validation Accuracy: {round(val_accuracy*100, 4)}%")
        self.metrics_dict['train_loss'] = self.train_losses
        self.metrics_dict['train_accuracy'] = self.train_accuracies
        self.metrics_dict['val_loss'] = self.val_losses
        self.metrics_dict['val_accuracy'] = self.val_accuracies

    def test(self):
        print(f"\nTesting:\n")
        self.test_accuracy, self.test_loss = self.eval_step(self.test_loader)
        if self.verbose:
            print(f"\tTest Loss: {round(self.test_loss, 4)}; Test Accuracy: {round(self.test_accuracy*100, 4)}%")
        self.metrics_dict['test_loss'] = self.test_loss
        self.metrics_dict['test_accuracy'] = self.test_accuracy
    
    def __get_scheduler_metric(self, train_accuracy, train_loss, val_accuracy, val_loss):
        return(val_accuracy)

    def train_step(self, data_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):
            image, label = data
            image = image.to(self.device)
            label = label.to(self.device)
        
            self.optimizer.zero_grad()
            output = self.model(image)
            self.loss = self.criterion(output, label)

            train_loss += self.loss.item()

            pred = torch.max(output.data, 1)[1]
            cur_correct = (pred == label).sum().item()
            cur_loss = self.loss.item()

            self.loss.backward()

            self.optimizer.step()
            
            total += label.size(0)
            correct += cur_correct
            train_loss += cur_loss

        train_accuracy = correct/total
        train_loss = train_loss/len(data_loader)

        return(train_accuracy, train_loss)
        
    def eval_step(self, data_loader):
        self.model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):
            image, label = data
            image = image.to(self.device)
            label = label.to(self.device)
                    
            output = self.model(image)
            loss = self.criterion(output, label)

            pred = torch.max(output.data, 1)[1]
            cur_correct = (pred == label).sum().item()
            cur_loss = loss.item()
                
            total += label.size(0)
            correct += cur_correct
            eval_loss += cur_loss

        eval_accuracy = correct/total
        eval_loss = eval_loss/len(data_loader)
        
        return(eval_accuracy, eval_loss)
    
    def use_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self):
        torch.save(self.model.state_dict(), self.file_name)

class ZigZagLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, mode='min', up_factor=1.1, down_factor=0.8, up_patience=10, down_patience=10, verbose=True):
        super(ZigZagLROnPlateau).__init__()
        self.optimizer = optimizer
        self.mode = mode
        self.up_factor = 1 + up_factor
        self.down_factor = 1 - down_factor
        self.up_patience = up_patience
        self.down_patience = down_patience
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.best_metric = np.Inf if self.mode == 'min' else -np.Inf
        self.verbose = verbose

    def step(self, metric):
        if self.mode == 'min':
            if metric < self.best_metric:
                self.best_metric = metric
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
                if self.num_good_epochs > self.up_patience:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")
                    self.num_good_epochs = 0
            else:
                self.num_bad_epochs += 1
                self.num_good_epochs = 0
                if self.num_bad_epochs > self.down_patience:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")
                    self.num_bad_epochs = 0
        else:
            if metric > self.best_metric:
                self.best_metric = metric
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
                if self.num_good_epochs > self.up_patience:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")
                    self.num_good_epochs = 0
            else:
                self.num_bad_epochs += 1
                self.num_good_epochs = 0
                if self.num_bad_epochs > self.down_patience:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")
                    self.num_bad_epochs = 0