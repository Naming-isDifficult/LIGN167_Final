from WaveNet.Layers import CausalConv1D, ResidualStack, DenseNet
import torch.nn as nn
import torch
import torch.optim as optim

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
       receptive_field = (2 ** layer_per_stack) * stack_size (might be wrong)
Also, the use of GPU will be guaranteed if possible.
'''
class WaveNetModel(nn.Module):
    def __init__(self, stack_size, layer_per_stack, input_dim, output_dim):
        #input_dim should be 1, however, it is flexible for text input
        #output_dim is the same as num_possible_values, by default its 256
        super(WaveNetModel, self).__init__()

        self.receptive_field = (2**layer_per_stack)*stack_size

        self.causal_conv = CausalConv1D(input_dim, output_dim)
        self.residual_blocks = ResidualStack(stack_size, layer_per_stack, output_dim)
        self.dense = DenseNet(output_dim)

        #The use of GPU has already been guaranteed by ResidualStack
        #we only need to check causal_conv and dense
        if torch.cuda.is_available():
            self.causal_conv = self.causal_conv.cuda()
            self.dense = self.dense.cuda()

    def forward(self, x):
        
        output = self.causal_conv(x)
        output = self.residual_blocks(output)
        #print(output)
        output = self.dense(output)
        
        return output

    def get_receptive_field(self):
        return self.receptive_field


'''
Actual application.
Assuming the input has shape (batch_size, input_dim, length) where input_dim should be 1
Flexibility of input_dim is kept for text input
The output should have shape (batch_size, num_possible_values, 1)
Since it's a softmax distribution, loss funcion will be set to cross-entrophy
'''
class WaveNet:
    def __init__(self, stack_size, layer_per_stack, input_dim, output_dim, lr=0.001):
        #input_dim should be 1, however, it is flexible for text input
        #output_dim is the same as num_possible_values, by default its 256
        
        self.model = WaveNetModel(stack_size, layer_per_stack, input_dim, output_dim)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss() if not torch.cuda.is_available()\
                    else nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.receptive_field = self.model.get_receptive_field()

    def train(self, input, target):
        #input.shape=(batch_size, input_dim, length)
        #target.shape=(batch_size, num_possible_values)
        #return the loss of this step

        self.optimizer.zero_grad()

        prediction = self.model(input) #prediction.shape(batch_size, num_possible_values, 1)
        prediction = prediction.squeeze(dim=2) #to fufill requirement of cross-entrophy
                                          #pred and target should have shape (batch_size, num_possible_values)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        return loss.data

    def get_receptive_field(self):
        return self.receptive_field