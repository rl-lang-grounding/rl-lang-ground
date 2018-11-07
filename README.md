# Attention Based Natural Language Grounding by Navigating Virtual Environment
Tensorflow code for our **WACV 2019** paper 

## 1) Usage
This code is based on [TensorFlow](https://www.tensorflow.org/). Please install it by following the instructions mentioned in TF website. [moviepy](https://pypi.python.org/pypi/moviepy) is a pre-requisite for storing GIFs.

## 2) Code organization
main.py is the main code containing the implementation of the architecture described in this paper. 

game_Env.py is the code for the new customizable 2D environment introduced in this paper. 

objects.json contains the specification of number and types of objects/obstacles. 

objects_new.json contains new increased number of objects. 

generateSentence.py - generates the feasible sentences for a given episode based on the environment configuration. 

vocab.txt - lists the possible unique words present in instructions. 

vocab_new.txt - increased vocabulary with new objects and new words present in instructions.

gifs directory contains some GIFs that were saved when we trained our attention architecture with n=5.

images directory contains the images used to represent agent, different objects and obstacles.

## 3) How to train?
Our implementation can be trained on a GPU. Please specify the GPU using CUDA_VISIBLE_DEVICES flag.
```
CUDA_VISIBLE_DEVICES="1" python main.py 
```
## 4) How to generate Attention Maps?
Run generateAttentionGifs.py to generate multiple gifs corresponding to the evolution of game state as well as of different attention maps in an episode. Extract out any single frame from those gifs(we used Preview app in mac). Once the images have been extracted, edit the location of the images(original and the attention map) in the Mask_Map.py code and run it to get the final mask. A sample extracted image of original state(orig.png), attention map(1.png) and the corresponding mask(1_masked.png) have also been uploaded.  

## 5) Sample depiction of trained agent
Natural language instruction is: "There are multiple green apple. Go to larger one."

First gif shows the agent's trajectory as it navigates to the large green apple. 
Second gif shows the egocentric view as observed by the agent at every step in its trajectory. 

![1](https://github.com/rl-lang-grounding/rl-lang-ground/raw/master/gifs/There_are_multiple_green_Apple_Go_to_larger_oneimage_13.gif)
![2](https://github.com/rl-lang-grounding/rl-lang-ground/raw/master/gifs/There_are_multiple_green_Apple_Go_to_larger_oneimage_13_ego.gif)

gifs directory contain additional GIFs that were saved when we trained our attention architecture with n=5.

## 6) 3D results 

To replicate the 3D results go this link https://github.com/rl-lang-grounding/DeepRL-Grounding

## 7) Acknowledgement
A3C implementation is based on open source implementation of [Arthur Juliani](https://github.com/awjuliani/DeepRL-Agents)


