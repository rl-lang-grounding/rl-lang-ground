# Learning to navigate by distilling visual information and natural language instructions
Tensorflow code for ICLR 2018 submission https://openreview.net/forum?id=HJPSN3gRW&noteId=HJPSN3gRW

## 1) Usage
This code is based on [TensorFlow](https://www.tensorflow.org/). Please install it by following the instructions mentioned in TF website. [moviepy](https://pypi.python.org/pypi/moviepy) is a pre-requisite for storing GIFs.

## 2) Code organization
main.py is the main code containing the implementation of the architecture described in this paper. 

game_Env.py is the code for the new customizable 2D environment introduced in this paper. 

objects.json contains the specificiation of number and types of objects/obstacles. 

generateSentence.py - generates the feasible sentences for a given episode based on the environment configuration. 

vocab.txt - lists the possible unique words present in instructions. 

gifs directory contains some GIFs that were saved when we trained our attention architecture with n=5.

images directory contains the images used to represent agent, different objects and obstacles.

## 3) How to train?
```
CUDA_VISIBLE_DEVICES="1" python main.py 
```

## 4) Acknowledgement
A3C implementation is based on open source implementation of [Arthur Juliani](https://github.com/awjuliani/DeepRL-Agents)

