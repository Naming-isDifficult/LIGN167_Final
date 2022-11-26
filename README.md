# LIGN167_Final
Final project for LIGN167 ---- a pytorch implementation for WaveNet.
Although in theory, since wavenet can build long-term dependency, it can also be used to handle text generation tasks. However, due to limited time, I'm afraid I might not be able to finish both parts. Text generation will be in the **TO-DO** list and all python files needed for text generation will be created. This markdown file will focus on audio part.

<br>

# Acknowledgement
 - WaveNet: A Generative Model for Raw Audio: https://arxiv.org/abs/1609.03499<br>
 - A TensorFlow implementation of DeepMind's WaveNet paper: https://github.com/ibab/tensorflow-wavenet

Code for mu-law encoding and decoding are based on ibab's tensorflow implementation. I use numpy to re-implement them.

<br>

# Requirements
* python >= 3.8
   - python 3.7 should also work but not guaranteed, the code is tested on python 3.8.3
* pytorch
   - If you have an nVidia GPU, please install pytorch-cuda, cuda and cudnn to enable operations on GPUs. They are much faster than operations on CPUs.
* librosa
* numpy


<br>

# Data Preprocess
The data preprocess will be divided into two parts:<br>
1. ~~Seperate a sound file into several shorter sound file~~ (Replaced by using yield to return inputs and outputs)
2. Create a torch.DataLoader to feed the model

For the input data, since librosa will produce normalized data (normalized to -1~1), it might be a good practice to use them directly. However, as for now, I cannot find a way to pass the value to DataLoader, so as for now, it is a part of my *TO-DO* list.

<br>

# WaveNet Model
## Layers Used by Model
1. Causal Convolution 1D (WaveNet.Layers.CausalConv1D)
    - Basic causal convolution layer
    - Has similar behavior to regular Conv1d layer except for causality is preserved here
2. Dilated Causal Convolution 1D (WaveNet.Layers.DilatedConv1D)
    - Supports dilation in order to expand receptive field faster
    - Other part is similar to Causal Convolution 1D
3. Residual Block (WaveNet.Layers.ResidualBlock)
    - Contains one Dialated Causal Convolution 1D layer and gated units similar to PixelCNN
    - For detailed information, please refer to https://arxiv.org/abs/1609.03499. ASCII art is a little bit hard in markdown.
4. Residual Stack (WaveNet.Layers.ResidualStack)
    - A stack of Residual Blocks
    - Receptive field can expand even faster when stacking stacks of residual blocks
5. Dense (WaveNet.Layers.DenseNet)
    - Contains softmax layer for classfication
    - 1x1 convolution is used instead of dense layer / fully-connected later
## Model Structure
Input -> Causal Convolution 1D (dimension will be raised to the same dimension as residual block) -> Resodial Stacks (actual output is skip connections) -> Dense -> result

<br>

# Causality
WaveNet is an autoregressive model based on CNN. In my code, causality  is preserved by only adding paddings at the beginning of the input. In this way, not only we can make input and output have the same length, but also make sure for each time step Xt, it won't contain information from future time step.
## Example
&ensp; input = [[[1 2 3 4 5 6 7 8]],<br>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; [[8 7 6 5 4 3 2 1]]] <br>
&ensp; Suppose all weights are 1 <br>
&ensp; Suppose bias is 0 <br>
&ensp; input_padding = [[[0 1 2 3 4 5 6 7 8]], <br>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; [[0 8 7 6 5 4 3 2 1]]]<br>
&ensp; output = [[[1 3 5 7 9 11 13 15]],<br>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp;[[8 15 13 11 9 7 5 3]]]

<br>

# Issues
1. **Skip connections might be wrong.**<br>
   The paper for WaveNet ***does not*** specify how to slice skip connections. According to ibab's Tensorflow implementation, only the last element (or last few elements) will be sliced into skip connections. <br>
   However, I don't really think that make sense. It's true for the last residual block (or last few residual blocks), last elements (or last few elements) do contain the information of the whole receptive field but that is not always the case ---- for the first residual block, for example, it only contains the information from last two elements. <br>
   I use a mean approach to generate skip connections, but this might be wrong. 

<br>

# To-DO
 - ~~Audio data loader~~ *(Now Functioning)*
   - Using original data directly instead of mu-law encoded or one-hot encoded as input *(Still Pending)*
 - ~~Audio data splitter~~
   - *Replaced by using yield instead of returning a batch created by the whole sound file.* 
 - ~~Network Layers~~ *(Now Functioning)*
 - Model
 - Data
 - Training
 - Generating samples
 - Text generation
