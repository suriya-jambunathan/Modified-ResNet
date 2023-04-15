import argparse
import torch
import yaml

from data import Data
from resnet import BasicBlock, ResNet
from train_test import TrainTest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Modified-ResNet', 
                                     description='train, test')
    parser.add_argument('--train', action = 'store_true', help = 'Argument to enable training model')
    parser.add_argument('--num_epochs', type = int, help = 'Argument to specify number of epochs to train')
    parser.add_argument('--test', action = 'store_true', help = 'Argument to enable testing model')
    parser.add_argument('--cuda', action = 'store_true', help = 'Argument to enable CUDA usage')
    args = parser.parse_args()

    # Loading the configuration file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initializing data, model, and train configurations
    data_config = config['data']
    model_config = config['model'][config['model']['use_model']]
    train_config = config['train']

    # Fetching Data
    data = Data(data_config)
    train_loader, val_loader, test_loader = data.get_data()

    # Initializing model
    model = ResNet(BasicBlock, model_config)

    if args.cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"Using {device.upper()}\n")

    # Initializing the TrainTest object
    train_test_obj = TrainTest((train_loader, val_loader, test_loader), model, train_config, device=torch.device('cuda'), verbose=True)

    if args.train:
        train_test_obj.num_epochs = args.num_epochs
        train_test_obj.train()
        train_test_obj.save_model()
    
    else:
        train_test_obj.use_model(config['model']['model_store']['best_model'])
    
    if args.test:
        train_test_obj.test()