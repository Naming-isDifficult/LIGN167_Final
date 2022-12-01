from WaveNet.Model import WaveNet
from DataLoader.AudioDataLoader import mu_law_decode
import os
import librosa
import torch
import numpy as np
import copy
from tqdm import tqdm
import soundfile

class Generator:
    '''
    Load an existing wavenet model to create a generator
    '''
    def __init__(self, path_to_model, num_possible_values=256, sr=16000):

        self.wavenet_model = WaveNet.load_model_pickle(path_to_model)

        self.sr = sr
        self.num_possible_values = num_possible_values
        self.receptive_field = self.wavenet_model.get_receptive_field()

    '''
    Change sample rate after initializing a generator
    '''
    def set_sr(self, sr=16000):
        self.sr = sr

    '''
    slice a sample for the model to predict
    expecting param seed is a numpy.ndarray with shape (length,)
    it will return a torch.tensor object for the model
    the use of cuda is guaranteed
    '''
    def slice_sample_to_pred(self, seed):
        re = None

        if len(seed) > self.receptive_field:
            re = seed[-self.receptive_field:]
        else:
            re = copy.deepcopy(seed)

        re = re.reshape((1,1,-1)) #(batch_size, input_dim, length)
        re = torch.tensor(re)
        re = re.type(torch.FloatTensor)

        return re if not torch.cuda.is_available()\
                  else re.cuda()

    '''
    Generate a sample using given model
    target_length is the length of output audio, in seconds
    path_to_output is the place for you to store your output wav file
    seed is the seed for audio. It has a default None value, in other words, if seed
    is not specified, a random number will be used.
    seed should be a numpy.ndarray object
    '''
    def generate_samples(self, target_length, path_to_output, seed=None):
        if seed is None:
            seed = np.random.random((1,))*255
            seed = np.cast['int32'](seed)
            seed = mu_law_decode(seed, self.num_possible_values)

        if len(seed) == 0:
            raise ValueError('The given seed is empty')
        
        result = seed #starting this point, seed will be used for samples feeding the model
                      #and result is used for storing results
        target_length = target_length * self.sr

        for i in tqdm(range(target_length-len(result))):

            seed = self.slice_sample_to_pred(result)

            next_sample = self.wavenet_model.pred(seed).item()
            next_sample = mu_law_decode(next_sample)

            result = np.append(result, next_sample)
        
        soundfile.write(path_to_output, result, self.sr)
        
