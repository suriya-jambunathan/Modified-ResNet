# Importing required libraries
import ast
import torch
import torchvision

class Data():
    """
    Class to facilitate fetching CIFAR10 Data, and do further processing,
    and finally return train, test, and validation loaders

    Methods
    -------
    get_data()
        Returns train, test, and validation data loaders.

    """
    def __init__(self, data_config):
        """
        Parameters
        ----------
        data_config: dict
            Data configuration file that contains important information
            about validation split, batch size, data augmentations
            specifications, and normalization.
        """
        self.dataset = data_config['name']
        self.val_split = data_config['val_split']
        self.batch_size = data_config['batch_size']
        self.augmentations = data_config['augmentations']
        self.normalization = data_config['normalization']

    def get_data(self):
        """
        Function to fetch CIFAR10 Data and adds Data Augmentations, 
        and necessary normalizations to return
        train, test, and validation data loaders.

        Returns
        -------
        train_loader: torch.utils.data.DataLoader
            Training Set Data Loader
        val_loader: torch.utils.data.DataLoader
            Validation Set Data Loader
        test_loader: torch.utils.data.DataLoader
            Testing Set Data Loader
        """
        # Augmentation Transformations 
        aug_transformations = []
        for augmentation in list(self.augmentations.keys()):
            aug_params = self.augmentations[augmentation]
            if augmentation == 'random_crop':
                aug_transformations.append(torchvision.transforms.RandomCrop(size=aug_params['size'], 
                                                                         padding=aug_params['padding']))
            elif augmentation == 'random_horizontal_flip':
                aug_transformations.append(torchvision.transforms.RandomHorizontalFlip(p=aug_params['p']))
            elif augmentation == 'random_resized_crop':
                aug_transformations.append(torchvision.transforms.RandomResizedCrop(size=aug_params['size'], 
                                                                                    scale=ast.literal_eval(aug_params['scale']), 
                                                                                    ratio=ast.literal_eval(aug_params['ratio'])))
        
        # Default Transformations
        default_transformations = []
        default_transformations.append(torchvision.transforms.ToTensor())
        default_transformations.append(torchvision.transforms.Normalize(mean=ast.literal_eval(self.normalization['mean']), 
                                                                        std=ast.literal_eval(self.normalization['std'])))

        # Training Set Transformations
        train_transformations = []
        train_transformations.extend(aug_transformations)
        train_transformations.extend(default_transformations)
        transform_train = torchvision.transforms.Compose(train_transformations)

        # Testing Set Transformations
        test_transformations = []
        test_transformations.extend(default_transformations)
        transform_test = torchvision.transforms.Compose(test_transformations)

        # Downloading CIFAR10 Dataset
        if self.dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # Split the Train Data into Train and Validation Sets
        val_size = int( self.val_split * len(trainset))
        train_size = len(trainset) - val_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

        # Generating the Data Loaders
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return(train_loader, val_loader, test_loader)