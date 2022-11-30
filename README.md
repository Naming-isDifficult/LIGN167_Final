# LIGN167_Final
Final project for LIGN167 ---- a pytorch implementation for WaveNet.
Although in theory, since wavenet can build long-term dependency, it can also be used to handle text generation tasks. However, due to limited time, I'm afraid I might not be able to finish both parts. Text generation will be in the **TO-DO** list and all python files needed for text generation will be created. This markdown file will focus on audio part.

<br>

# Acknowledgement
 - WaveNet: A Generative Model for Raw Audio: https://arxiv.org/abs/1609.03499<br>
 - A TensorFlow implementation of DeepMind's WaveNet paper: https://github.com/ibab/tensorflow-wavenet

Code for mu-law encoding and decoding are based on ibab's tensorflow implementation. I use numpy to re-implement them.<br>
I also use ibab's tensorflow implementation for help when I reach skip connection and receptive field. However, I can't fully agree with ibab's implementation, though I could be, and most likely am, wrong. For more information, check **Issues** section.

<br>

# Requirements
* python >= 3.8
   - python 3.7 should also work but not guaranteed, the code is tested on python 3.8
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
### Example
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
1. **Skip connections**<br>
   The paper for WaveNet ***does not*** specify how to slice skip connections. According to ibab's Tensorflow implementation, only the last element (or last few elements) will be sliced into skip connections. <br>
   However, I don't really think that make sense. It's true for the last residual block (or last few residual blocks), last elements (or last few elements) do contain the information of the whole receptive field but that is not always the case ---- for the first residual block, for example, it only contains the information from last two elements. <br>
   I use a mean approach to generate skip connections, but this might be wrong. 
2. **Receptive Field**<br>
   Although the paper does mention how to calculate the receptive field of one residual block and how to stack residual blocks together, the way to calculate receptive field of stack of residual blocks is not mentioned. In the tensorflow implementation by ibab, the receptive field is the sum of all dilations plus one. However, I'm currently suspecting it might be one of the following:
    - (2 ** layer_per_stack - 1) * stack_size + 1 (ibab's implementation)
    - (2 ** layer_per_stack) ** stack_size (unsure)
    - (2 ** layer_per_stack) * stack_size (my implementation)<br>
3. **Saving/Loading Models**<br>
    Although pytorch does provide torch.save() and torch.load() methods to save/load models, I don't really want to use they due to uncertainty ---- I don't have enough time to make sure everything is working. Instead, I use another approach ---- saving the model with python moudule pickle. To save/load your own model, save_model_pickle() and load_model_pickle() methods are recommended. They are implemented based on picke module. For save_model() and load_model() methods, they are implemented based on putorch and their functionalities *are not* guaranteed.

<br>

# To-DO
 - ~~Audio data loader~~ *(Now Functioning)*
   - Using original data directly instead of mu-law encoded or one-hot encoded as input *(Still Pending)*
 - ~~Audio data splitter~~
   - *Replaced by using yield instead of returning a batch created by the whole sound file.* 
 - ~~Network Layers~~ *(Now Functioning)*
 - ~~Model~~ *(Now Functioning)*
 - Use a file to store hyperparameters instead of hard coding it while training
 - Data
 - Training
 - Generating samples
 - Text generation
