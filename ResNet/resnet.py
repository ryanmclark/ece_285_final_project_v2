#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:08:55 2018

@author: Ryan Clark

Header:
This file defines the class to utilize the ResNet architecture for semantic
segmentation. It currently supports both the 18 and 34 layer architectures.
You can initialize your ResNet  with a single parameter as such:
  net = ResNet(layers = XX)
where XX is either 18 or 34. From this point the, network expects a single
Torch autograd variable of Torch Float Tensor shape (N, C, H, W), where
  N = number of number of samples
  C = number of channels
  H = height of Images
  W = width of Images
The following call with train the network,
  yinit = net(data)
where yinit will be the result from the foward propigation through the ResNet
encoder and decoder.

The ResNet class is defined by the layers input parameter.
Since the ResNet archiecture is broken down into groups of repeating blocks of
one of four different feature space dimensions, it is apparent that the only
difference between 18 layers and 34 layers is the number of repeated blocks
within a group. This being said, the ResNet layers parameter merely selects
the list corresponding to the number of block repetitions within each group.

The structure of the ResNet class is broken down into subclasses ResNet_Encoder
and ResNet_Decoder. While the ResNet class defines the network structure as well
as the propigation through the ResNet_Encoder and Res_Net_Decoder, the
ResNet_Encoder and ResNet_Decoder classes define the group strucutre as well as
the propigation through Encoder_Group and Decoder_Group(s) respectively. This
secondary subclasses define the structure and propigation through each block
within the group.

**Note only works for Color Images. To use for GrayScale change input output
feature dimension to 1**
"""

# Libraries
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# Global Variables
# Used to keep track of the indices from the the initial max pool layer for the
# decoder's unpool layer of the network.
global mp_idx
mp_idx = None

class ResNet(nn.Module):

    def __init__(self, layers):
        super(ResNet, self).__init__()

        self.layers = layers

        if self.layers == 18:
            self.blocks_per_group = [2, 2, 2, 2]
        elif self.layers == 34:
            self.blocks_per_group = [3, 4, 6, 3]

        self.Encoder = ResNet_Encoder(self.blocks_per_group)
        self.Decoder = ResNet_Decoder(self.blocks_per_group)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)

        return x


class ResNet_Encoder(nn.Module):

    def __init__(self, blocks_per_group):
        super(ResNet_Encoder, self).__init__()

        self.bpg = blocks_per_group

        # Parameters:
        # Encoder_Group(in_channels, out_channels, num_of_blocks, beginning)
        self.group_1 = Encoder_Group(3, 64, self.bpg[0], True)
        self.group_2 = Encoder_Group(64, 128, self.bpg[1])
        self.group_3 = Encoder_Group(128, 256, self.bpg[2])
        self.group_4 = Encoder_Group(256, 512, self.bpg[3])

    def forward(self, x):
        x = self.group_1.forward(x)
        x = self.group_2.forward(x)
        x = self.group_3.forward(x)
        x = self.group_4.forward(x)

        return x


class Encoder_Group(nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_of_blocks, beginning = False):
        super(Encoder_Group, self).__init__()

        # Relevant Parameters:
        # nn.Conv2d(in_channels, out_channels, kernel_size,
        #           stride, padding,... bias)
        # nn.BatchNorm2d(channels)
        # nn.ReLU()
        # nn.MaxPool2d(kernel_size, stride, return_indices)

        # Handles the initial 7 by 7 convolution

        # Defines the number of repeating blocks within the group
        self.num_of_blocks = num_of_blocks
        # Used to trigger initial group of the Encoder (7 by 7 conv group)
        self.beginning = beginning

        self.initial_conv = nn.Conv2d(in_channels, out_channels,
                                      kernel_size = 7, stride = 2, bias = False)

        # The first block of each group includes a stride as well as an increase
        # in feature space dimensionality. Thus the shortcut becomes a 1 by 1
        # convolution as well to address these needs.
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                                 stride = 2, padding = 1, bias = False)
        self.in_shortcut_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size = 1, stride = 2, bias = False)

        # The most repeated convolution in each group.
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                                  padding = 1, bias = False)

        # Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function
        self.relu = nn.ReLU()
        # Aggregation function if defined for the inital group
        self.max_pool = nn.MaxPool2d(kernel_size = (3, 3), stride = 2,
                                     return_indices = True)

    def _initial_group(self, x):
        # The initial group that involves the pooling layer and 7 by 7 conv is
        # unique. This addresses the requirements for the group.

        # 7 by 7 conv, followed by batchnorm, and max pooling
        x = self.initial_conv(x)
        x = self.bn(x)
        global mp_idx
        x, mp_idx = self.max_pool(x)

        # The next group following holds a constant dimension in the feature
        # space. It is also unique. Loops throuth each block in the group.
        for cnt in range(self.num_of_blocks):
            shortcut = x

            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.bn(x)
            x += shortcut
            x = self.relu(x)

        return x

    def _standard_group(self, x):
        # The most common group structure in the network, handles the inital
        # downsampling / feature channel expansion, then the common blocks.

        shortcut = self.in_shortcut_conv(x)
        shortcut = self.bn(shortcut)

        x = self.in_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x += shortcut
        x = self.relu(x)

        for cnt in range(self.num_of_blocks - 1):
            shortcut = x

            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.bn(x)
            x += shortcut
            x = self.relu(x)

        return x

    def forward(self, x):

        if self.beginning:
            x = self._initial_group(x)
        else:
            x = self._standard_group(x)

        return x

class ResNet_Decoder(nn.Module):

    def __init__(self, blocks_per_group):
        super(ResNet_Decoder, self).__init__()

        self.bpg = blocks_per_group

        # Parameters:
        # Decoder_Group(in_channels, out_channels, num_of_blocks,
        #               end, output_padding)
        self.group_1 = Decoder_Group(512, 256, self.bpg[3], False, 0)
        self.group_2 = Decoder_Group(256, 128, self.bpg[2], False, 1)
        self.group_3 = Decoder_Group(128, 64, self.bpg[1], False, 1)
        self.group_4 = Decoder_Group(64, 3, self.bpg[0], True, 0)

    def forward(self, x):
        x = self.group_1.forward(x)
        x = self.group_2.forward(x)
        x = self.group_3.forward(x)
        x = self.group_4.forward(x)

        return x


class Decoder_Group(nn.Module):

    def __init__(self, in_channels, out_channels, num_of_blocks, end = False,
                 output_padding = 0):
        super(Decoder_Group, self).__init__()

        # Relevant Parameters:
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        #           padding,... bias)
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        #                    stride, padding, output_padding,... bias)
        # nn.BatchNorm2d(channels)
        # nn.ReLU()
        # nn.Softmax2d()
        # nn.MaxPool2d(kernel_size, stride, return_indices)

        # Number of repetitions in each group.
        self.num_of_blocks = num_of_blocks
        # Used to trigger the final group
        self.end = end
        # Some upsampling need additional padding for correct dimensions.
        self.output_padding = output_padding

        # Used for the most common convolution in each block
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3,
                              padding = 1, bias = False)

        # Used for the convolutions leaving the group where upsampling happens.
        # Sometimes additional padding is needed for the output to match the
        # respective input.
        self.out_conv = nn.ConvTranspose2d(in_channels, out_channels,
                    kernel_size = 3, stride = 2, padding = 1,
                    output_padding = self.output_padding, bias = False)
        self.out_shortcut_conv = nn.ConvTranspose2d(in_channels, out_channels,
                    kernel_size = 1, stride = 2,
                    output_padding = self.output_padding, bias = False)

        # Used for the final group with the 7 by 7 conv and the unpooling layer.
        self.final_conv = nn.ConvTranspose2d(in_channels, 2,
                kernel_size = 7, stride = 2, output_padding = 1, bias = False)

        # Batch Normalization
        self.in_bn = nn.BatchNorm2d(in_channels)
        self.out_bn = nn.BatchNorm2d(out_channels)
        # Activation Functions
        self.relu = nn.ReLU()
        # Unpooling function if defined for the final group
        self.max_unpool = nn.MaxUnpool2d(kernel_size = (3, 3), stride = 2)
        # Final classification
        #self.softmax = nn.Softmax2d()

    def _standard_group(self, x):
        # Defines the most common block structure each group.

        for cnt in range(self.num_of_blocks - 1):
            shortcut = x

            x = self.conv(x)
            x = self.in_bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.in_bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.in_bn(x)
            x += shortcut
            x = self.relu(x)

        # Final block per group where upsampling is needed.
        shortcut = self.out_shortcut_conv(x)
        shortcut = self.out_bn(shortcut)

        x = self.conv(x)
        x = self.in_bn(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.in_bn(x)
        x = self.relu(x)

        x = self.out_conv(x)
        x = self.out_bn(x)
        x += shortcut
        x = self.relu(x)

        return x

    def _final_group(self, x):
        # The final group of the Decoder

        for cnt in range(self.num_of_blocks):
            shortcut = x

            x = self.conv(x)
            x = self.in_bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.in_bn(x)
            x = self.relu(x)

            x = self.conv(x)
            x = self.in_bn(x)
            x += shortcut
            x = self.relu(x)

        global mp_idx
        desired_shape = torch.Size([mp_idx.shape[0], 3, mp_idx.shape[2]*2 + 2,
                                                        mp_idx.shape[3]*2 + 2])
        x = self.max_unpool(x, mp_idx, output_size = desired_shape)
        x = self.final_conv(x) # 7 by 7 conv
        #x = self.softmax(x)
        
        return x

    def forward(self, x):

        if not self.end:
            x = self._standard_group(x)
        else:
            x = self._final_group(x)

        return x
