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
        Returns Optimizer.

    get_scheduler(scheduler_id, scheduler_config)
        Returns Scheduler.

    train()
        Trains and validates the model based on the configuration.

    test()
        Tests the trained model.

    train_step(data_loader)
        Training function defining operations within one epoch.
    
    eval_step(data_loader)
        Evaluation function defining operations within one epoch.

    use_model(path)
        Loads the model state dictionary from the specified path.
    
    save_model()
        Stores the trained model state dictionary.
    """
    def __init__(self, data, model, train_config, device, verbose = True):
        """
        Parameters
        ----------
        data: tuple of torch.utils.data.DataLoader
            train_loader, val_loader, test_loader

        model: torch.nn.Module
            ResNet Model

        train_config: dict
            Train Configuration
        
        device: torch.device
            CPU or CUDA
        
        verbose: bool
            Whether to display to console.
        """

        # Train, Validation and Test Loaders
        self.train_loader, self.val_loader, self.test_loader = data

        # Moving the model to device
        self.model = model.to(device)

        # Verbose
        self.verbose = verbose

        # Number of Epochs
        self.num_epochs = train_config['num_epochs']

        # Setting the Criterion
        self.criterion = self.get_criterion(train_config['criterion'])

        # Setting the Optimizer
        self.optimizer = self.get_optimizer(train_config['use_optimizer'], 
                                            train_config['optimizers'][train_config['use_optimizer']])
        
        # Setting the Scheduler
        self.scheduler = self.get_scheduler(train_config['use_scheduler'], 
                                            train_config['schedulers'][train_config['use_scheduler']])
        
        # Device
        self.device = device

        # Initializing self arrays for tracking metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Metrics Dictionary
        self.metrics_dict = {'train_loss': None, 'train_accuracy': None, 
                             'val_loss': None, 'val_accuracy': None,
                             'test_loss': None, 'test_accuracy': None}
        
        # File name to store the model
        self.file_name = train_config['model_store']['session_model']

    def get_criterion(self, criterion_id):
        """
        Function to get the Criterion.

        Parameters
        ----------
        criterion_id: str
            Criterion Name
        
        Returns
        -------
        Criterion: torch.nn.CrossEntropyLoss()
        """
        # CrossEntropyLoss
        if criterion_id == 'cross_entropy_loss':
            return(torch.nn.CrossEntropyLoss())

    def get_optimizer(self, optimizer_id, optimizer_config):
        """
        Function to get the Optimizer.

        Parameters
        ----------
        optimizer_id: str
            Optimizer Name
        
        optimizer_config: dict
            Optimizer Configuration
        
        Returns
        -------
        Optimizer: torch.optim.SGD()
        """
        # SGD
        if optimizer_id == 'sgd':
            return(torch.optim.SGD(self.model.parameters(), lr=optimizer_config['lr'], 
                                   momentum=optimizer_config['momentum'], 
                                   weight_decay=optimizer_config['weight_decay'], 
                                   nesterov=optimizer_config['nesterov']))

    def get_scheduler(self, scheduler_id, scheduler_config):
        """
        Function to get the Scheduler.

        Parameters
        ----------
        scheduler_id: str
            Scheduler Name
        
        scheduler_config: dict
            Scheduler Configuration
        
        Returns
        -------
        Scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau()/ ZigZagLROnPlateau()
        """
        # ReduceLROnPlateau
        if scheduler_id == 'reduce_lr_on_plateau':
            return(torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=scheduler_config['mode'], 
                                                              factor=scheduler_config['factor'], 
                                                              patience=scheduler_config['patience'], verbose=self.verbose))
        
        # ZigZagLROnPlateau
        elif scheduler_id == 'zigzag_lr_on_plateau':
            return(ZigZagLROnPlateau(mode=scheduler_config['mode'], 
                                     up_factor=scheduler_config['up_factor'], down_factor=scheduler_config['down_factor'],
                                     up_patience=scheduler_config['up_patience'], down_patience=scheduler_config['down_patience']))

    def train(self):
        """
        Function to train and validate the model based on the configuration.
        """
        print(f"\nTraining for {self.num_epochs} epochs:\n")
        for epoch in range(self.num_epochs):

            # Training the model
            train_accuracy, train_loss = self.train_step(self.train_loader)

            # Validating the model
            val_accuracy, val_loss = self.eval_step(self.val_loader)

            # Computing the Metric to pass to Scheduler
            metric = self.__get_scheduler_metric(train_accuracy, train_loss, val_accuracy, val_loss)

            # Updating the metric to scheduler to update learning rate
            self.scheduler.step(metric)

            # Appending accuracy and loss metrics to self array
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)

            # Console Out
            if self.verbose:
                print(f"\n\tEpoch: {epoch+1}/{self.num_epochs}")
                print(f"\tTraining Loss: {round(train_loss, 4)}; Training Accuracy: {round(train_accuracy*100, 4)}%")
                print(f"\tValidation Loss: {round(val_loss, 4)}; Validation Accuracy: {round(val_accuracy*100, 4)}%")
        
        # Updating accuracy and loss metrics to self metrics dictionary
        self.metrics_dict['train_loss'] = self.train_losses
        self.metrics_dict['train_accuracy'] = self.train_accuracies
        self.metrics_dict['val_loss'] = self.val_losses
        self.metrics_dict['val_accuracy'] = self.val_accuracies

    def test(self):
        """
        Function to test the model based on the configuration.
        """
        print(f"\nTesting:\n")

        # Testing the model
        self.test_accuracy, self.test_loss = self.eval_step(self.test_loader)

        # Console Out
        if self.verbose:
            print(f"\tTest Loss: {round(self.test_loss, 4)}; Test Accuracy: {round(self.test_accuracy*100, 4)}%")

        # Updating accuracy and loss metrics to self metrics dictionary
        self.metrics_dict['test_loss'] = self.test_loss
        self.metrics_dict['test_accuracy'] = self.test_accuracy
    
    def __get_scheduler_metric(self, train_accuracy, train_loss, val_accuracy, val_loss):
        """
        Function to get the metric to pass to scheduler.
        
        Parameters
        ----------
        train_accuracy: float
            Train Accuracy
        
        train_loss: float
            Train Loss

        val_accuracy: float
            Validation Accuracy

        val_loss: float
            Validation Loss

        Returns
        -------
        metric: float
            val_accuracy
        """
        return(val_accuracy)

    def train_step(self, data_loader):
        """
        Function defining training operations within one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        Returns
        -------
        train_accuracy: float
            Train Accuracy
        
        train_loss: float
            Train Loss
        """
        # Setting model mode to train
        self.model.train()

        # Initializing variables to facilitate metrics calculation.
        train_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):

            # Images and corresponding Labels
            image, label = data
            image = image.to(self.device)
            label = label.to(self.device)

            # Clearing Optimizer Gradients
            self.optimizer.zero_grad()

            # Forward Pass
            output = self.model(image)

            # Computing Loss
            self.loss = self.criterion(output, label)

            # Computing Accuracy
            pred = torch.max(output.data, 1)[1]
            cur_correct = (pred == label).sum().item()
            cur_loss = self.loss.item()

            # Backward Pass
            self.loss.backward()
            self.optimizer.step()
            
            # Updating loss and accuracy
            total += label.size(0)
            correct += cur_correct
            train_loss += cur_loss

        # Computing final Accuracy and Loss
        train_accuracy = correct/total
        train_loss = train_loss/len(data_loader)

        return(train_accuracy, train_loss)
        
    def eval_step(self, data_loader):
        """
        Function defining evaluation operations within one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        Returns
        -------
        eval_accuracy: float
            Evaluation Accuracy
        
        eval_loss: float
            Evaluation Loss
        """
        # Setting model mode to eval
        self.model.eval()

        # Initializing variables to facilitate metrics calculation.
        eval_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):

            # Images and corresponding Labels
            image, label = data
            image = image.to(self.device)
            label = label.to(self.device)
            
            # Forward Pass
            output = self.model(image)

            # Computing Loss
            loss = self.criterion(output, label)

            # Computing Accuracy
            pred = torch.max(output.data, 1)[1]
            cur_correct = (pred == label).sum().item()
            cur_loss = loss.item()
            
            # Updating loss and accuracy
            total += label.size(0)
            correct += cur_correct
            eval_loss += cur_loss

        # Computing final Accuracy and Loss
        eval_accuracy = correct/total
        eval_loss = eval_loss/len(data_loader)
        
        return(eval_accuracy, eval_loss)
    
    def use_model(self, path):
        """
        Function to load the model state dictionary from the specified path.

        Parameters
        ----------
        path: str
            Model State Dictionary File Path (.pth)
        """
        self.model.load_state_dict(torch.load(path))

    def save_model(self):
        """
        Function to store the trained model state dictionary.
        """
        torch.save(self.model.state_dict(), self.file_name)


class ZigZagLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    """
    Class defining the ZigZag Learning Rate Scheduler

    Methods
    -------
    step(metric)
        Updates the learning rate of the Optimizer based on the metric.
    """
    def __init__(self, optimizer, mode='min', up_factor=1.1, down_factor=0.8, up_patience=10, down_patience=10, verbose=True):
        """
        Parameters
        ----------
        optimizer: torch.optim
            Model Optimizer

        mode: str
            Whether to minimize or maximize the metric.

        up_factor: float
            Factor by which the learning rate will be scaled up.

        down_factor: float
            Factor by which the learning rate will be scaled down.

        up_patience: int
            Number of epochs to wait before scaling up.

        down_patience: int
            Number of epochs to wait before scaling down.
        
        verbose: bool
            Whether to print to console output.
        """
        super(ZigZagLROnPlateau).__init__()

        # Model Optimizer
        self.optimizer = optimizer

        # Minimize or Maximize
        self.mode = mode

        # lr_new = lr*(1 + up_factor)
        self.up_factor = 1 + up_factor

        # lr_new = lr(1 - down_factor)
        self.down_factor = 1 - down_factor

        # Epochs to wait before upscaling
        self.up_patience = up_patience

        #Epochs to wait before downsizing
        self.down_patience = down_patience

        # Number of bad epochs (decrease)
        self.num_bad_epochs = 0

        # Number of good epochs (increase)
        self.num_good_epochs = 0

        # Setting default best_metric to +ve or -ve infinity depending on mode
        self.best_metric = np.Inf if self.mode == 'min' else -np.Inf

        # Verbose
        self.verbose = verbose

    def step(self, metric):
        """
        Function to update the learning rate based on the metric.

        Parameters
        ----------
        metric: float
            Performance Metric
        """
        # Metric to be Minimized
        if self.mode == 'min':

            # If Current metric better than Best Metric
            if metric < self.best_metric:

                # Update Best Metric
                self.best_metric = metric

                # Reset Bad Epochs and Increment Good Epochs
                self.num_bad_epochs = 0
                self.num_good_epochs += 1

                # If number of good epochs is greater than the up_patience
                if self.num_good_epochs > self.up_patience:

                    # Update the learning rate by upscaling
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr

                    # Console Out
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset Number of Good Epochs
                    self.num_good_epochs = 0

            # If Current metric didn't improve over the Best Metric
            else:

                # Reset Good Epochs and Increment Bad Epochs
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

                # If number of bad epochs is greater than the down_patience
                if self.num_bad_epochs > self.down_patience:

                    # Update the learning rate by downsizing
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr

                    # Console Out
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset Number of Good Epochs
                    self.num_bad_epochs = 0

        # Metric to be Maximized
        else:

            # If Current metric better than Best Metric
            if metric > self.best_metric:

                # Update Best Metric
                self.best_metric = metric

                # Reset Bad Epochs and Increment Good Epochs
                self.num_bad_epochs = 0
                self.num_good_epochs += 1

                # If number of good epochs is greater than the up_patience
                if self.num_good_epochs > self.up_patience:

                    # Update the learning rate by upscaling
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr

                    # Console Out
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset Number of Good Epochs
                    self.num_good_epochs = 0

            # If Current metric didn't improve over the Best Metric
            else:

                # Reset Good Epochs and Increment Bad Epochs
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

                # If number of bad epochs is greater than the down_patience
                if self.num_bad_epochs > self.down_patience:

                    # Update the learning rate by downsizing
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr

                    # Console Out
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset Number of Good Epochs
                    self.num_bad_epochs = 0