# Learning to navigate by distilling visual information and natural language instructions
Tensorflow code for ICLR 2018 submission https://openreview.net/forum?id=HJPSN3gRW&noteId=HJPSN3gRW

## 1) Usage
This code is based on [TensorFlow](https://www.tensorflow.org/). Please install it by following the instructions mentioned in TF website. [moviepy](https://pypi.python.org/pypi/moviepy) is a pre-requisite for storing GIFs.

## 2) Code organization
main.py is the main code containing the implementation of the architecture described in this paper. 

game_Env.py is the code for the new customizable 2D environment introduced in this paper. 

objects.json contains the specification of number and types of objects/obstacles. 

generateSentence.py - generates the feasible sentences for a given episode based on the environment configuration. 

vocab.txt - lists the possible unique words present in instructions. 

gifs directory contains some GIFs that were saved when we trained our attention architecture with n=5.

images directory contains the images used to represent agent, different objects and obstacles.

## 3) How to train?
Our implementation can be trained on a GPU. Please specify the GPU using CUDA_VISIBLE_DEVICES flag.
```
CUDA_VISIBLE_DEVICES="1" python main.py 
```

## 4) Sample depiction of trained agent
Natural language instruction is: "There are multiple green apple. Go to larger one."

First gif shows the agent's trajectory as it navigates to the small green apple. 
Second gif shows the egocentric view as observed by the agent at every step in its trajectory. 

![1](https://github.com/rl-lang-grounding/rl-lang-ground/raw/master/gifs/There_are_multiple_green_Apple_Go_to_larger_oneimage_13.gif)
![2](https://github.com/rl-lang-grounding/rl-lang-ground/raw/master/gifs/There_are_multiple_green_Apple_Go_to_larger_oneimage_13_ego.gif)

gifs directory contain additional GIFs that were saved when we trained our attention architecture with n=5.

## 5) Acknowledgement
A3C implementation is based on open source implementation of [Arthur Juliani](https://github.com/awjuliani/DeepRL-Agents)


