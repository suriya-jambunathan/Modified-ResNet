# Modified-ResNet

## Introduction
The aim of this project is to optimize results using
the ResNet architecture and a comprehensive hyper-
parameter tuning method. The architecture is strictly
limited to 5 million parameters, without pre-trained
weights, and only using the CIFAR-10 dataset. The
goal is to achieve high efficiency and effectiveness with-
out compromising computational efficiency. The focus
on hyper-parameter tuning allows for optimal param-
eter settings. The project aims to develop a cutting-
edge deep learning model that delivers exceptional per-
formance while adhering to strict computational con-
straints. By leveraging ResNet architecture and apply-
ing thorough hyper-parameter tuning, the model can
offer unparalleled accuracy and efficiency, making it a
valuable asset for various applications


## Architecture

![plot](./home/shubham/Downloads/zigzag_resnet.png.png?raw=true "Title")

## Results

After conducting a series of experiments, we have iden-
tified that the most optimal model for our task is the
ZigZag ResNet architecture. The model is trained using
a batch size of 32 and a combination of augmentations
including RandomCrop with a size of 32 and padding of
4, RandomHorizontalFlip with a probability of 0.5, and
RandomResizedCrop with a size range of 32 and scale
ratio of 0.8-1.0 and an aspect ratio range of 0.8-1.2.
For optimization, we have used the SGD optimizer
with a learning rate of 0.01, momentum of 0.8, weight
decay of 0.0005, and Nesterov acceleration. To further
improve the performance of the model, we have incorpo-
rated the ReduceLROnPlateau scheduler with a factor
of 0.1 and a patience of 5.
The model’s performance has been evaluated through
multiple experiments, and the finalized configuration
has shown outstanding results. With the identified hy-
perparameters, our model has achieved an accuracy of
93% on the CIFAR-10 dataset. These findings suggest
that the ZigZag ResNet architecture, along with the
selected hyperparameters, is a powerful tool for image
classification tasks on the CIFAR-10 dataset.
