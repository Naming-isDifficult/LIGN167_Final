import torch
import torch.nn as nn
import numpy as np

#------ convolution layers section ------#
'''
Dialated Causal 1D Convolution Layer
Input and output should have the same dimension
For the length, I'm not 100% sure here, in theory we don't need to keep the length
since we only need the output containing all of the information
However, just in case I need that, a parameter called keep_dim is used here
If kip_dim is set to be True, the input and output will have the same length
If not, the length will be input_len-dilation
To make sure it's causal, paddings will only be added to the beginning
The amount of paddings should equals to dilation
Zero-paddings are used here
e.g. input = [[[1 2 3 4 5 6 7 8]],
              [[8 7 6 5 4 3 2 1]]]
     dilation = 4
     Suppose all weights are 1
     If keep_dim=True:
         input_padding = [[[0 0 0 0 1 2 3 4 5 6 7 8]],
                          [[0 0 0 0 8 7 6 5 4 3 2 1]]]
         output = [[[1 2 3 4 6 8 10 12]],
                   [[8 7 6 5 12 10 8 6]]]
    If keep_dim=False
         output=[[[6 8 10 12]],
                 [[12 10 8 6]]]
'''
class DilatedConv1D(nn.Module):
    
    def __init__(self, input_dim=1, dilation=1, keep_dim=True):
        super(DilatedConv1D, self).__init__()
        
        self.dilation = dilation
        self.keep_dim = keep_dim

        self.pad = nn.functional.pad #cuz we only have three dimensions,
                                     #built-in padding layers cannot be applied here
                                     #hence, because of the uncertainty of pytorch conv1d layer built-in padding
                                     #(mainly because I'm not familiar with pytorch)
                                     #we use nn.functional.pad instead
                                     #parameters should be (input, (self.dilation, 0),value=0)
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=input_dim,\
                              kernel_size=2,stride=1,padding=0,\
                              dilation=dilation) #input and output should have the same dimension
    
    def testing_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.fill_(1)
                layer.bias.data.fill_(0)

    def forward(self, x):
        
        if self.keep_dim:
            x = self.pad(x, (self.dilation, 0), value=0)
        
        output = self.conv(x)

        return output

'''
Causal 1D Convolution Layer
Its behavior is the same as DilatedConv1D with dilation=1 and keep_dim=True
The only difference is, in order to fit residual block which can have different dimensions,
output dimension of causal conv1d can be changed
e.g. input = [[[1 2 3 4 5 6 7 8]],
              [[8 7 6 5 4 3 2 1]]]
     Suppose all weights are 1
     If outputdim = 1:
         output = [[[1 3 5 7 9 11 13 15]],
                   [[8 15 13 11 9 7 5 3]]]
    If outputdim = 2:
         output=[[[1 3 5 7 9 11 13 15],
                  [1 3 5 7 9 11 13 15]],

                 [[8 15 13 11 9 7 5 3],
                  [8 15 13 11 9 7 5 3]]]
'''
class CausalConv1D(nn.Module):

    def __init__(self, input_dim=1, output_dim=512):
        super(CausalConv1D, self).__init__()

        self.pad = nn.functional.pad #cuz we only have three dimensions,
                                     #built-in padding layers cannot be applied here
                                     #hence, because of the uncertainty of pytorch conv1d layer built-in padding
                                     #(mainly because I'm not familiar with pytorch)
                                     #we use nn.functional.pad instead
                                     #parameters should be (input, (1, 0),value=0)
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim,\
                              kernel_size=2, stride=1, padding=0)
        
    def testing_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.fill_(1)
                layer.bias.data.fill_(0)

    def forward(self, x):

        x = self.pad(x, (1,0), value=0)
        output = self.conv(x)

        return output

#------ convolution layers section finished ------#

#------ resnet section ------#
#------ resnet section finished ------#

#------ dense section ------#
#------ dense section finished ------#