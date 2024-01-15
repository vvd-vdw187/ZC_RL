import torch.nn as nn

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1_bn(in_channels, out_channels, stride=1):
    """1x1 convolution with batch normalization"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )

def conv3x3_bn(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with batch normalization"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
    )

def conv1x1_bn_relu(in_channels, out_channels, stride=1):
    """1x1 convolution with batch normalization and ReLU"""
    return nn.Sequential(
        conv1x1_bn(in_channels, out_channels, stride=stride),
        nn.ReLU(inplace=True),
    )

def conv3x3_bn_relu(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with batch normalization and ReLU"""
    return nn.Sequential(
        conv3x3_bn(in_channels, out_channels, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )

def conv1x1_relu(in_channels, out_channels, stride=1):
    """1x1 convolution with ReLU"""
    return nn.Sequential(
        conv1x1(in_channels, out_channels, stride=stride),
        nn.ReLU(inplace=True),
    )

def conv3x3_relu(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with ReLU"""
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )

# Make Impala Base layer
