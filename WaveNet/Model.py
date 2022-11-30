from WaveNet.Layers import CausalConv1D, ResidualStack, DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import os
import pickle

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
    def __init__(self, stack_size, layer_per_stack, input_dim, res_dim, output_dim):
        #input_dim should be 1, however, it is flexible for text input
        #output_dim is the same as num_possible_values, which should be 256 for audio
        super(WaveNetModel, self).__init__()

        self.receptive_field = (2**layer_per_stack)*stack_size

        self.causal_conv = CausalConv1D(input_dim, res_dim)
        self.residual_blocks = ResidualStack(stack_size, layer_per_stack, res_dim, output_dim)
        self.dense = DenseNet(output_dim)

        #The use of GPU has already been guaranteed by ResidualStack
        #we only need to check causal_conv and dense
        if torch.cuda.is_available():
            self.causal_conv = self.causal_conv.cuda()
            self.dense = self.dense.cuda()

    def forward(self, x):
        
        output = self.causal_conv(x)
        output = self.residual_blocks(output)
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
    def __init__(self, stack_size, layer_per_stack, input_dim, res_dim, output_dim, lr=0.001):
        #input_dim should be 1, however, it is flexible for text input
        #output_dim is the same as num_possible_values, by default its 256
        
        self.model = WaveNetModel(stack_size, layer_per_stack, input_dim, res_dim, output_dim)
        self.loss = nn.CrossEntropyLoss() if not torch.cuda.is_available()\
                    else nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.receptive_field = self.model.get_receptive_field()
        self.default_dir_cache = None

    
    def get_receptive_field(self):
        return self.receptive_field

    def get_default_model_dir(self):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%Y-%m-%d')
        return 'Model/WaveNet_Model_{date}'.format(date=current_time)

    def get_model_name(self, step, loss):
        return 'wavenet_step{current_step}_loss{current_loss}.model'.format(current_step=step,\
                                                                            current_loss=loss)


    #------ methods for training ------#
    def train(self, input, target):
        #train one step
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


    def save_model(self, step, loss, model_dir=None, default_dir=True):
        #save model for future use
        #if default_dir is set to be True, it will be saved to ./Model/WaveNet_Model_Date
        #if default_dir is set to be False, it will be saved to model_dir

        if default_dir:
            if self.default_dir_cache is not None:
                model_dir = self.default_dir_cache
            else:
                self.default_dir_cache = self.get_default_model_dir()
        model_name = self.get_model_name(step, loss)
        path_to_model = os.path.join(model_dir, model_name)

        torch.save(self.model.state_dict(), path_to_model)
        print('Model saved to {dir}'.format(dir=path_to_model))


    def save_model_pickle(self, step, loss, model_dir=None, default_dir=True):
        #save model for future use
        #this method will use a pickle approach
        if default_dir:
            if self.default_dir_cache is not None:
                model_dir = self.default_dir_cache
            else:
                self.default_dir_cache = self.get_default_model_dir()
        model_name = self.get_model_name(step, loss)
        path_to_model = os.path.join(model_dir, model_name)

        pickle.dump(self, open(path_to_model, 'wb'))
        print('Model saved to {dir}'.format(dir=path_to_model))
    #------ methods for training finish ------#


    #------ methods for predicting ------#
    @staticmethod
    def load_model(path_to_model):
        #load a model saved by pytorch
        print('Model at {dir} loaded'.format(dir=path_to_model))
        return torch.load(open(path_to_model))

    @staticmethod
    def load_model_pickle(path_to_model):
        #load a model saved by pickle
        print('Model at {dir} loaded'.format(dir=path_to_model))
        return pickle.load(open(path_to_model, 'rb'))

    def pred(self, input):
        #predict next value
        #expecting input.shape=(1, input_dim, receptive_field)
        #however, receptive_field is not strictly restricted but recommended
        if input.shape[0] != 1:
            raise ValueError('Batch size is strictly restricted to 1 for generation,\
            but your batch size is {your_batch}'.format(your_batch=input.shape[0]))

        prediction = self.model(input) #expecting shape=(1, num_possible_values, 1)
        prediction = prediction.squeeze() #expecting shape=(num_possible_values)
        output = F.log_softmax(prediction, dim=0) #expecting shape=(num_possible_values)

        return output.argmax() #expecting shape=(1,)
    #------ methods for predicting finish ------#