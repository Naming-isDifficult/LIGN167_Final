import os
import librosa
import numpy as np
import torch
import torch.utils.data as data

#------ encoding-decoding section ------#
'''
Assuming data is an np.ndarray with shape (num_sample, 1), or at least an np.ndarray
Assuming num_possible_value is an integer representing how many different possible values (by default it's 256)
The return value should be an np.ndarray with shape (num_sample,num_possible_value)
'''
def one_hot_encode(data:np.ndarray, num_possible_value:int = 256) -> np.ndarray:

    possible_values = np.arange(num_possible_value)
    data = data.reshape((-1,1)) #just in case I miss something

    re = np.cast['float32'](data==possible_values)
    
    return re

'''
Assuming data is an np.ndarray
To enable batch operations (for DataLoader), input shape won't be assumed
Most likely it should be either (num_sample, num_possible_values) or (batch_size, num_sample, num_posible_values)
The return value should most likely be an np.ndarrat with shape (num_sample,) or (batch_size, num_sample)
'''
def one_hot_decode(data:np.ndarray, axis=1) -> np.ndarray:
    return data.argmax(axis)

'''
Assuming data is an np.ndarray with shape (num_sample,), it should be normalized to -1~1
Assuming num_possible_value is an integer representing how many different possible values (by default it's 256)
The return value should be an np.ndarray with shape (num_sample,)
Based on a tensorflow implementation: 
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py
tf.minimum will be ignored (since np.min doesn't support broadcasting)
'''
def mu_law_encode(data:np.ndarray, num_possible_value:int=256) -> np.ndarray:

    #check if data is normalized
    if (data>1.0).any() or (data<-1.0).any():
        raise ValueError('Normalize Data First')

    mu = float(num_possible_value-1)
    data_abs = np.abs(data)
    magnitude = np.log1p(mu*data_abs) / np.log1p(mu+1)
    signal = np.sign(data) * magnitude

    re = ((signal+1)/2 * mu +0.5)

    return np.cast['int32'](re)

'''
Assuming data is an np.ndarray with shape (num_sample,)
Assuming num_possible_value is an integer representing how many different possible values (by default it's 256)
The return value should be an np.ndarray with shape (num_sample,)
Based on a tensorflow implementation: 
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py
'''
def mu_law_decode(data:np.ndarray, num_possible_value:int=256) -> np.ndarray:

    mu = num_possible_value - 1
    signal = 2*(np.cast['float32'](data)/mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)

    re = np.sign(signal) * magnitude

    return re
#------ encoding-decoding section finished ------#


#------ data-loading section ------#
'''
Assuming file is an string representing the path of one audio file
Assuming sr is the sample rate of the audio file (by default it's 16000) (why 16000? cuz 16kHz is pretty common for mp3)
Since librosa supprots resampling, we don't really need to care about sampling that much
However, according to https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py,
resampling might cause error
In that case, the best practice might be using the sample rate of the audio file (which can be modified in other methods)
The return value should be an np.ndarray with shape (num_sample,)
The return value should be normalized to -1~1 (done by librosa)
'''
def load_audio_file(file:str, sr:int=16000, trim:bool=True) -> np.ndarray:
    
    data, _ = librosa.load(file, sr=sr, mono=True)
    if trim:
        data, _ = librosa.effects.trim(data)

    return data

'''
Sub-class of data.Dataset
It will iterate the folder containing sound files
However, program WILL NOT check if it is a sound file,
plz make sure all files under source folder are sound files
'''
class AudioDataset(data.Dataset):
    '''
    Assuming source_folder is a string representing the path to the folder with sound files
    Assuming sr is an integer representing the sample rate (by default it's 16000)
    Assuming num_possible_value is an integer representing how many different possible values (by default it's 256)
    Assuming trim is a boolean representing whether removing blank at the beginning and the end or not
    '''
    def __init__(self, source_folder:str='AudioData', sr:int=16000, num_possible_value:int=256, trim=True):
        super(AudioDataset, self).__init__()

        self.num_possible_value = num_possible_value
        self.sr = sr
        self.trim = trim
        self.source_folder = source_folder
        self.file_list = [x for x in os.listdir(source_folder)]

    '''
    Override
    Returning an np.ndarray representing an audio file at a time
    Data will be one-hot encoded
    However, this is mainly because I cannot find a way to take a value from a tuple while coding DataLoader
    The actual input data will be mu-law encoded instead of one-hot encoded
    Though yes, using the original data might be a better way
    '''
    def __getitem__(self, index):

        file_path = os.path.join(self.source_folder,\
                                 self.file_list[index])
        
        data = load_audio_file(file_path, self.sr, self.trim)
        mu_encoded_data = mu_law_encode(data, self.num_possible_value) #training data
        one_hot_encoded_data = one_hot_encode(mu_encoded_data, self.num_possible_value) #labels

        return one_hot_encoded_data

    '''
    Override
    Returnning the size of dataset
    '''
    def __len__(self):
        return len(self.file_list)

'''
Sub-class of data.DataLoader
It will generate training data and label pairs according to given params
'''
class AudioDataLoader(data.DataLoader):
    def __init__(self, receptive_field:int,\
                        dataset = None,\
                        source_folder:str='AudioData',\
                        batch_size:int=64,\
                        sr:int=16000,\
                        num_possible_value:int=256,\
                        trim=True):
        if dataset is None:
            dataset = AudioDataset(source_folder, sr,num_possible_value, trim)
        super(AudioDataLoader, self).__init__(dataset, 1, True) #True for shuffling

        self.receptive_field = receptive_field
        self.pseudo_batch = batch_size #batch NOT created by reading several sound file at a time
                                        #instead, it should be created from a single file
        self.collate_fn = self.generate_training_pairs
        self.has_gpu = torch.cuda.is_available()
    
    '''
    Equivalent to torch.from_numpy
    Autograd is guaranteed, use of gpu is guaranteed
    '''
    def numpy_to_variable(self, data: np.ndarray) -> torch.Tensor:

        tensor = torch.from_numpy(data).float()
        re = torch.autograd.Variable(tensor.cuda()) if self.has_gpu\
            else torch.autograd.Variable(tensor)

        return re

    '''
    Assuming data is an np.ndarray representing remaining data with shape (num_samples, num_possible_values)
    This method will calculate the actually batch size according to pseudo batch size and remaining data length
    '''
    def calculate_batch_size(self, data:np.ndarray):
        return self.pseudo_batch if len(data) > self.receptive_field + self.pseudo_batch\
            else len(data)-self.receptive_field

    '''
    Customized collate_fn
    This method will generate training pairs for one batch
    The first value returned by this method is input with shape (batch_size, input_dim, receptive_field)
    where input_dim is 1 (i.e. (batch_size, 1, receptive_field))
    The second value returned by this method is output with shape (batch_size, output_dim)
    '''
    def generate_training_pairs(self, stacked_input):
        #stacked_input.shape = (1, num_samples, num_possible_values)

        data = stacked_input[0] #data = (num_samples, num_possible_values), no paddings
                                #Also, num_possible_values are output dim, the inputdim should always be 1

        while len(data) > self.receptive_field:
            actual_batch_size = self.calculate_batch_size(data)

            #build targets for one batch
            targets = data[self.receptive_field:\
                        self.receptive_field+actual_batch_size,] #targets.shape=(batch_size, num_possible_values)
                                                                    #or, targets.shape=(batch_size, output_dim)
            #build inputs for one batch with stupid loop
            input_raw = data[:self.receptive_field+actual_batch_size-1]
            input_raw = one_hot_decode(input_raw).reshape((1,-1)) #input_raw.shape=(1, num_sample_need)
                                                                    #or, input_raw.shape=(input_dim, num_sample_need)
            input_list = []
            while len(input_list) < actual_batch_size:
                input_list.append(input_raw[:,len(input_list):\
                                len(input_list)+self.receptive_field])
            inputs = np.stack(input_list, axis=0) #inputs.shape=(batch_size, 1, receptive_field)
                                                    #or, inputs.shape=(batch_size, input_dim, receptive field)
            inputs = mu_law_decode(inputs) #normalize data to [-1,1] and also make sure 0 is actual 0
                                            #however, due to mu-law, 0 will become a number near 0 but not actual 0
                                            #so it should be better if we can use original signal as input
            yield self.numpy_to_variable(inputs),\
                  self.numpy_to_variable(targets)
            
            #preparing data for next batch
            data = data[actual_batch_size:,:]

#------ data-loading section finished ------#