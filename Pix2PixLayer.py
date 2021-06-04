import torch
import torch.nn as nn

#Generator의 인코더 블럭
def makeEncoderBlock(in_channels, out_channels, kernel_size=4, padding=1, stride=2, lrelu=0.2, batch_norm=True):
    layers = list()
    
    layers.append(nn.LeakyReLU(lrelu))
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    return nn.Sequential(*layers)

#Generator의 디코더 블럭
def makeDecoderBlock(in_channels, out_channels, kernel_size=4, padding=1, stride=2, batch_norm=True, dropout=0.5):
    layers = list()
    
    layers.append(nn.ReLU())
    layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

#Discriminator 블럭
def makeDiscriminatorBlock(in_channels, out_channels, kernel_size=4, padding=1, stride=2, lrelu=0.2, batch_norm=True):
    layers = list()
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    layers.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layers)