from WaveNet.Model import WaveNet
from DataLoader.AudioDataLoader import AudioDataset, AudioDataLoader
import os

class Trainer:
    '''
    data_source_folder is where the training data are, by defaut it's ./AudioData
    sr is samplerate of the songs
    num_possible_values, which is also output_dim, is for mu-law encoding
    trim represents if the program is deleting mute part at the beginning and the end
    It's highly recommended to set trim=True (which is the default value)
    '''
    def __init__(self, data_source_folder='AudioData',\
                 sr=16000, num_posible_values=256, trim=True):

        self.data_set = AudioDataset(data_source_folder, sr, num_posible_values, trim)
        self.data_loader = None
        self.wavenet_model = None
    
    '''
    Initialize the model with given parameters
    If the model has been initialized before, an error will be thrown
    stack_size represents how many residua stacks in the model
    layer_per_stack represents how many layers in one residual block
    input_dim represents the dimension of input (by default it should be 1)
    res_dim represents the dimension of residual blocks
    output_dim represents the dimension of output
    (which should be the same as num_possible_values)
    If suppress_error is set to True, original wavenet_model will be overwrited
    '''
    def initialize_model(self, batch_size, stack_size, layer_per_stack,\
                         input_dim, res_dim, output_dim, lr=0.001,\
                         suppress_error=False):

        if not suppress_error:
            if self.wavenet_model is not None:
                raise RuntimeError('You have already initialized this the model in this trainer.\
                If you want to overwrite it, please use suppress_error=True.')

        self.wavenet_model = WaveNet(stack_size, layer_per_stack,\
                                     input_dim, res_dim, output_dim, lr)
        self.wavenet_model.reset_default_model_dir()

        receptive_field = self.wavenet_model.get_receptive_field()

        self.data_loader = AudioDataLoader(receptive_field,\
                                           dataset=self.data_set,\
                                           batch_size=batch_size)

    '''
    Initialize the model by reading a pre-existing model
    If the model has been initialized before, an error will be thrown
    '''
    def load_model(self, batch_size, path_to_model, path_to_weight, suppress_error=False):

        if not suppress_error:
            if self.wavenet_model is not None:
                raise RuntimeError('You have already initialized this the model in this trainer.\
                If you want to overwrite it, please use suppress_error=True.')
        
        self.wavenet_model = WaveNet.load_model(path_to_model, path_to_weight)
        self.wavenet_model.reset_default_model_dir()
        
        receptive_field = self.wavenet_model.get_receptive_field()

        self.data_loader = AudioDataLoader(receptive_field,\
                                           dataset=self.data_set,\
                                           batch_size=batch_size)

    '''
    To take data from dataloader
    '''
    def get_data(self):
        for data in self.data_loader:
            for inputs, outputs in data:
                yield inputs, outputs

    '''
    Train the model and save the model
    max_epoch means maximum times of iterating the whole dataset.
    However, due to the nature of waveform, most likely it is impossible to finish even
    one epoch
    model_dir is the target directory for model, however, it will be overwrite if default_dir
    is set to be True (which is default)
    steps_to_save represents how many steps will be trained before saving a model
    maximum_model represents there will be at most {maximum_model} models saved in target directory
    '''
    def train_model(self, max_epoch=5, model_dir=None, default_dir=True,\
                    steps_to_save=10, maximum_model=5):

        step = 0
        if default_dir:
            model_dir = self.wavenet_model.get_default_model_dir()

        for i in range(max_epoch):
            for inputs, outputs in self.get_data():
                step = step+1
                loss = self.wavenet_model.train(inputs, outputs)
                loss = round(loss.item(), 4)
                print('Step: {st}, Loss: {lo}'.format(st=step, lo=loss))

                if(step%steps_to_save == 0):
                    self.wavenet_model.save_model(step, loss,\
                                                  model_dir, default_dir)
                    #check how many models have been saved
                    file_list = os.listdir(model_dir)
                    if(len(file_list) > maximum_model*2):
                        file_list.sort(key=lambda x: int(x.split('_')[1][4:]))
                        os.remove(os.path.join(model_dir, file_list[0])) #remove model
                        os.remove(os.path.join(model_dir, file_list[1])) #remove weight

        #save the model after all epoches are done
        self.wavenet_model.save_model(step, loss.item(),\
                                      model_dir, default_dir)
        #check how many models have been saved
        file_list = os.listdir(model_dir)
        if(len(file_list) > maximum_model):
            file_list.sort()
            os.remove(os.path.join(model_dir, file_list[0]))

                    
