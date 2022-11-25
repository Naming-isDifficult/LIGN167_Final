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

For the input data, since librosa will produce normalized data (normalized to -1~1), it might be a good practice to use them directly. However, as for now, I cannot find a way to pass the value to DataLoader, so as for now, it is a part of my *TO-DO* list.

<br>

# To-DO
 - ~~Audio data loader~~ *(Now Functioning)*
   - Using original data directly instead of mu-law encoded or one-hot encoded as input *(Still Pending)*
 - ~~Audio data splitter~~
   - *Replaced by using yield instead of returning a batch created by the whole sound file.* 
 - Network Layers
 - Model
 - Data
 - Training
 - Generating samples
 - Text generation
