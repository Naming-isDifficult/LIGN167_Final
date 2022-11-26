from WaveNet.Layers import ResidualStack, DenseNet
import torch.nn as nn

'''
The whole WaveNet model
Assuming input has shape (batch_size, input_dim, length)
where input_dim should be 1, length should equal to receptive field
The model will automatically calculate receptive field while initializing according
to given stack size and layers per stack
e.g.
    stack = 5
    layers = 8
    -> dilation_each_stack=[1,2,4,8,16,32,64,128]
       overall_dilation=[1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,\
                         1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,\
                         1,2,4,8,16,32,64,128]
       receptive_field = (2 ** layer_per_stack) ** stack_size (might be wrong)
'''
class WaveNetModel(nn.Module):
    def __init__(self, stack_size, layer_per_stack, output_dim):
        #output_dim is the same as num_possible_values, by default its 256
        super(WaveNetModel, self).__init__()
