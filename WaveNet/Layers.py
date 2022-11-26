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

    def __init__(self, input_dim, output_dim):
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
'''
Residual block used by WaveNet model, for more information, plz check:
https://arxiv.org/pdf/1609.03499.pdf
I'm not 100% sure about this
I believe both residual output and skip connection output have the same dimension
However, if that is the case, since there is not other layers changing the dimension,
both residual dimension and skip connection dimention has to be the same as output dimension
Although I implement residual block like that (e.g. residual_input_dim = num_possible_values, 256 by default),
I might still be wrong ---- they both can be part of hyperparameters.
ATTENTION:
Currently skip connection part might be wrong
How to slice skip connectin IS NOT specified in the paper for WaveNet
According to https://github.com/ibab/tensorflow-wavenet/blob/master/,
a tensorflow implementation of WaveNet which has been cited for many times,
the way to slice skip connection is
    skip_cut = tf.shape(out)[1] - output_width
    out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
which basically means only the last ouput_width (in my code output_width is always 1)
elements will be selected.
However, I don't really think it makes sense. For more information, plz check README
Here I use a mean approach. Each block will caculate the mean of all samples.
'''
class ResidualBlock(nn.Module):

    def __init__(self, residual_input_dim, dilation):
        super(ResidualBlock, self).__init__()

        #convolution layers, including one dilated causal convolution layer,
        #one regular convolution layer for residual output and skip connection output
        self.dilated_conv = DilatedConv1D(residual_input_dim,\
                                          dilation=dilation,\
                                          keep_dim=True)
        self.regular_conv = nn.Conv1d(in_channels=residual_input_dim,\
                                      out_channels=residual_input_dim,\
                                      kernel_size=1) #parameter names added for readability

        #PixelCNN gate unit, as mentioned in the paper
        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()


    def forward(self, x):

        #dilated causal conv
        dilated_output = self.dilated_conv(x)

        #gate units
        tanh_gated_x = self.gate_tanh(dilated_output)
        sigmoid_gated_x = self.gate_sigmoid(dilated_output)
        gate_output = tanh_gated_x * sigmoid_gated_x

        #1x1 convolution
        regular_conv_x = self.regular_conv(gate_output)

        #skip connection
        #shape = (batch_size, num_possible_values, 1)
        skip_connection_output = torch.mean(regular_conv_x, 2, True)

        #residual
        #shape = (batch_size, num_possible_values, length)
        residual_output = regular_conv_x + x 

        return residual_output, skip_connection_output

'''
A stack of residual blocks used by WaveNet.
This can also be initialized directly while initializing models
(e.g. adding a list of residual blocks)
However, in order to increase readability and make sure each residual block
is gpu-ready, this class is created
It will automatically calculate dilation according to given stack size and layers per stack
Dimension for residual blocks should always be the same as num_possible_values
e.g.
    stack = 5
    layers = 8
    -> dilation_each_stack=[1,2,4,8,16,32,64,128]
       overall_dilation=[1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,\
                         1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,\
                         1,2,4,8,16,32,64,128]
'''
class ResidualStack(nn.Module):

    def __init__(self, stack_size, layer_per_stack, residual_input_dim):
        super(ResidualStack, self).__init__()

        self.has_gpu = torch.cuda.is_available()

        #check residual stack building section for the code
        #self.residual_blocks should be a list of residual blocks
        self.residual_blocks = self.get_block_list(stack_size,\
                                                   layer_per_stack,\
                                                   residual_input_dim)

    def forward(self, x):
        input_for_next_block = x #the input for next residual block is the output
                                 #of previous block
        skip_connections = [] #actual output we are going to use

        for block in self.residual_blocks:
            input_for_next_block, skip_connection = block(input_for_next_block)
            skip_connections.append(skip_connection)

        output = torch.stack(skip_connections)
        return torch.sum(output, dim=0) #add all skip connections together
                                        #output shape should be (batch_size, num_possible_values, 1)

    #------ residual stack building section ------#
    def get_one_block(self, residual_input_dim, dilation):
        
        block = ResidualBlock(residual_input_dim, dilation)

        #if GPU is available, the use of GPU is guaranteed
        return block if not self.has_gpu\
                     else block.cuda()


    def get_block_list(self, stack_size, layer_per_stack, residual_input_dim):
        
        block_list = []

        dilation_each_stack = [2**i for i in range(layer_per_stack)]

        for i in range(stack_size):
            for dilation in dilation_each_stack:
                block = self.get_one_block(residual_input_dim, dilation)
                block_list.append(block)

        return block_list
    #------ residual stack building section finished

#------ resnet section finished ------#

#------ dense section ------#
'''
Last layer of WaveNet, responsible for classification
It will take the output from residual stack (which is the sum of skip connections) as input
and will output the classification result based on softmax
The output should have shape (batch_size, num_possible_values, 1)
For more information, plz check:
https://arxiv.org/pdf/1609.03499.pdf
'''
class DenseNet(nn.Module):

    def __init__(self, input_dim):
        #input and output should have the same dim
        super(DenseNet, self).__init__()
        
        self.relu0 = nn.ReLU() #this will be applied directly to input

        self.conv1 = nn.Conv1d(input_dim, input_dim, 1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(input_dim, input_dim, 1)
        self.softmax = nn.Softmax(dim=1) #input.shape=(batch_size, num_possible_values, 1)

    def forward(self, x):
        output = self.relu0(x)

        output = self.conv1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.softmax(output)

        return output

#------ dense section finished ------#