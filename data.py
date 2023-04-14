import torch
import torchvision

class Data():
    def __init__(self, data_config):
        self.dataset = data_config['name']
        self.val_split = data_config['val_split']
        self.batch_size = data_config['batch_size']
        self.augmentations = data_config['augmentations']
        self.normalization = data_config['normalization']

    def get_data(self):

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
                                                                                scale=aug_params['scale'], 
                                                                                ratio=aug_params['ratio']))
        
        default_transformations = []
        default_transformations.append(torchvision.transforms.ToTensor())
        default_transformations.append(torchvision.transforms.Normalize(mean=self.normalization['mean'], std=self.normalization['std']))

        train_transformations = []
        train_transformations.extend(aug_transformations)
        train_transformations.extend(default_transformations)
        transform_train = torchvision.transforms.Compose(train_transformations)

        test_transformations = []
        test_transformations.extend(default_transformations)
        transform_test = torchvision.transforms.Compose(test_transformations)
    
        if self.dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # Split the train data into train and validation sets
        val_size = int( self.val_split * len(trainset))
        train_size = len(trainset) - val_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return(train_loader, val_loader, test_loader)