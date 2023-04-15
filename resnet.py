# Importing Required Libraries
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Class to define the Basic Building Block of the ResNet Architecture.

    Attributes
    ----------
    expansion: int

    Methods
    -------
    forward(x)
        Returns the output tensor after applying the BasicBlock Layers
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        """
        Parameters
        ----------
        in_planes: int
            Shape of the input planes
        planes: int
            Shape of the output planes
        stride: int
            Defines Skip Connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        Function defining the BasicBlock PyTorch Class.
        
        Attributes
        ----------
        x: torch.tensor
            Input Tensor

        Returns
        -------
        out: torch.tensor
            Output Tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    """
    Class to define the custom ResNet Architecture defined by the configuration.

    Methods
    -------
    forward(x)
        Returns the output tensor after applying the ResNet Layers
    """
    def __init__(self, basicblock, model_config):
        """
        Parameters
        ----------
        basicblock: BasicBlock
            BasicBlock Class object
        model_config: dict
            Model Configuration
        """
        super(ResNet, self).__init__()
        num_classes = model_config['num_classes']
        blocks = model_config['blocks']
        block_keys = list(blocks.keys())
        self.in_planes = blocks[block_keys[0]]['planes']

        # First Layer
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Block Layers
        layers = []
        for block_key in block_keys:
            block = blocks[block_key]
            planes = block['planes']
            num_blocks = block['count']
            stride = block['stride']
            strides = [stride] + [1]*(num_blocks-1)
            cur_layer = []
            for stride in strides:
                cur_layer.append(basicblock(self.in_planes, planes, stride))
                self.in_planes = planes * basicblock.expansion
            layers.extend(cur_layer)
        self.model_blocks = nn.Sequential(*layers)

        # Final Layer
        self.linear = nn.Linear(blocks[block_keys[-1]]['planes']*basicblock.expansion, num_classes)

    def forward(self, x):
        """
        Function defining the ResNet PyTorch Class.
        
        Attributes
        ----------
        x: torch.tensor
            Input Tensor

        Returns
        -------
        out: torch.tensor
            Output Tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.model_blocks(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out