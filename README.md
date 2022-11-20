# LIGN167_Final
Final project for LIGN167 ---- a pytorch implementation for WaveNet.
Although in theory, since wavenet can build long-term dependency, it can also be used to handle text generation tasks. However, due to limited time, I'm afraid I might not be able to finish both parts. Text generation will be in the **TO-DO** list and all python files needed for text generation will be created. This markdown file will focus on audio part.

<br>

# Acknowledgement
 - WaveNet: A Generative Model for Raw Audio: https://arxiv.org/abs/1609.03499<br>
 - A TensorFlow implementation of DeepMind's WaveNet paper: https://github.com/ibab/tensorflow-wavenet

Code for mu-law encoding and decoding are based on ibab's tensorflow implementation. I use numpy to re-implement them.

<br>

# Data Preprocess
The data preprocess will be divided into two parts:<br>
1. Seperate a sound file into several shorter sound file
2. Create a torch.DataLoader to feed the model

For the input data, it's true that we should be able to use raw data read by librosa since they have been normalized to -1~1. However, I still decide to use one-hot encoding version.

<br>

# To-DO
 - Audio data loader
 - Audio data splitter
 - Network Layers
 - Model
 - Data
 - Training
 - Generating samples
 - Text generation
