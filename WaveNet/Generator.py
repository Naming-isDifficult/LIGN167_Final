from WaveNet.Model import WaveNet
from DataLoader.AudioDataLoader import mu_law_decode
import os
import librosa
import torch
import numpy as np
import copy
from tqdm import tqdm

class Generator:
    '''
    Load an existing wavenet model to create a generator
    '''
    def __init__(self, path_to_model, num_possible_values=256, sr=16000):

        self.wavenet_model = WaveNet.load_model_pickle(1, path_to_model)
        # set batch_size to one, though it won't be used in this module

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

        re = torch.tensor(re)

        return re if not torch.cuda.is_available()\
                  else re.cuda()

    '''
    Generate a sample using given model
    target_length is the length of output audio, in seconds
    seed is the seed for audio. It has a default None value, in other words, if seed
    is not specified, a random number will be used.
    '''
    def generate_samples(self, target_length, seed=None):
        if seed is None:
            seed = np.random.random.((1,))*255
            seed = np.cast['int32'](seed)
            seed = mu_law_decode(seed, self.num_possible_values)

        if len(seed) == 0:
            raise ValueError('The given seed is empty')
        
        result = seed #starting this point, seed will be used for samples feeding the model
                      #and result is used for storing results
        target_length = target_length * self.sr

        for i in tqdm(range(target_length-len(result))):

            seed = self.slice_sample_to_pred(result)

            next_sample = self.wavenet_model.(seed).item()
            next_sample = mu_law_decode(next)

            result = np.append(result, next_sample)
        
