import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
from collections import deque
import random
import os
import sys

from random import choice
from time import sleep
from time import time

from game_Env import Env as gameEnv
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope

flags = tf.app.flags
flags.DEFINE_string("model_path",'./model/',"the directory where the models are going to be saved")
flags.DEFINE_string("gif_path",'./attentionFrames/',"the directory where the gifs are going to be stored")
flags.DEFINE_string("json_file",'objects.json',"the json file descibing the environment")
flags.DEFINE_string("vocab_file",'vocab.txt',"the file containing vocabulary words")
flags.DEFINE_integer("max_episode_length", 100, "maximum length of episode")
flags.DEFINE_integer("num_workers", 1, "number of threads")

FLAGS = flags.FLAGS

def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder


def process_frame(frame):
	s = scipy.misc.imresize(frame,[84,84])
	s = np.reshape(s,[np.prod(s.shape)]) 
	return s


def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def PReLU(x, init=0.001, name='output'):
	init = tf.constant_initializer(init)
	with variable_scope.variable_scope(None, 'PReLU', [x]) as sc:
		alpha = variables.model_variable('alpha', [], initializer=init)
		x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
		ret = tf.multiply(x, 0.5, name=name)
	return ret

class AC_Network():
	def __init__(self,s_size,a_size,scope):
		with tf.variable_scope(scope):
			#Input and visual encoding layers
			self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
			self.reward_desc = tf.placeholder(shape= [None,9,40],dtype = tf.float32)

			self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,3])
			self.imageIn = tf.cast(self.imageIn, tf.float32) / 255.0
			################################################################################################################
			self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
			self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
			self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
			self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
			self.lr = tf.placeholder(shape=[],dtype = tf.float32)

			net = self.imageIn
			with tf.device('/gpu:0'):
				with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(),
											weights_regularizer=slim.l2_regularizer(0.0005)):
					
					net = slim.conv2d(net,kernel_size=[5,5], num_outputs=32, stride=[2,2],padding='VALID', activation_fn=None)
					net = PReLU(net)
					
					net = slim.conv2d(net,kernel_size=[5,5], num_outputs=32, stride=[2,2],padding='VALID', activation_fn=None)
					net = PReLU(net)
					
					net = slim.conv2d(net,kernel_size=[4,4],num_outputs=64, stride=[1,1],padding='VALID', activation_fn=None)
					net = PReLU(net)
					
					net = slim.conv2d(net,kernel_size=[3,3],num_outputs=64, stride=[2,2],padding='VALID', activation_fn=None)
					net = PReLU(net)
					# print net.get_shape()
					reward_unrolled = tf.unstack(self.reward_desc,self.reward_desc.get_shape()[1],1)	
					gru_cell = tf.contrib.rnn.GRUCell(16)
					outputs,_ = tf.contrib.rnn.static_rnn(gru_cell,reward_unrolled,dtype=tf.float32)
					reward_out = outputs[-1]
					
					self.reward_out = reward_out
					reward_out1 = slim.fully_connected(reward_out,64,activation_fn=None)
					reward_out1 = PReLU(reward_out1)
					reward_out1 = tf.reshape(reward_out1, [1,1,64,1])
					attention1 = tf.nn.conv2d(net, reward_out1, strides=[1, 1, 1, 1], padding='VALID')
					
					reward_out2 = slim.fully_connected(reward_out,64,activation_fn=None)
					reward_out2 = PReLU(reward_out2)
					reward_out2 = tf.reshape(reward_out2, [1,1,64,1])
					attention2 = tf.nn.conv2d(net, reward_out2, strides=[1, 1, 1, 1], padding='VALID')
					
					reward_out3 = slim.fully_connected(reward_out,64,activation_fn=None)
					reward_out3 = PReLU(reward_out3)
					reward_out3 = tf.reshape(reward_out3, [1,1,64,1])
					attention3 = tf.nn.conv2d(net, reward_out3, strides=[1, 1, 1, 1], padding='VALID')

					reward_out4 = slim.fully_connected(reward_out,64,activation_fn=None)
					reward_out4 = PReLU(reward_out4)
					reward_out4 = tf.reshape(reward_out4, [1,1,64,1])
					attention4 = tf.nn.conv2d(net, reward_out4, strides=[1, 1, 1, 1], padding='VALID')

					reward_out5 = slim.fully_connected(reward_out,64,activation_fn=None)
					reward_out5 = PReLU(reward_out5)
					reward_out5 = tf.reshape(reward_out5, [1,1,64,1])
					attention5 = tf.nn.conv2d(net, reward_out5, strides=[1, 1, 1, 1], padding='VALID')

					hidden = tf.concat([attention1, attention2, attention3, attention4, attention5], 3)
					self.attention_map = hidden
					print('hidden shape after concat ',hidden.get_shape())
					hidden = slim.conv2d(hidden, kernel_size=[3,3], num_outputs=64, stride=[1,1], activation_fn=None)
					hidden = PReLU(hidden)
					hidden = slim.conv2d(hidden, kernel_size=[3,3], num_outputs=64, stride = [1,1], activation_fn=None)
					hidden = PReLU(hidden)
					hidden = slim.flatten(hidden)
					


					lstm_cell = tf.contrib.rnn.BasicLSTMCell(32,state_is_tuple=True)
					c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
					h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
					self.state_init = [c_init, h_init]
					c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
					h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
					self.state_in = (c_in, h_in)
					rnn_in = tf.expand_dims(hidden, [0])
					state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
					lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
					lstm_cell, rnn_in, initial_state=state_in, time_major=False)
					lstm_c, lstm_h = lstm_state
					self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
					rnn_out = tf.reshape(lstm_outputs, [-1, 32])

					policy = slim.fully_connected(rnn_out,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
					value = slim.fully_connected(rnn_out,1,activation_fn=None,biases_initializer=None)
				
			self.policy=policy
			self.value=value
			
			if scope != 'global':
				
				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
				
				entropy_beta = tf.get_variable('entropy_beta', shape=[],
									   initializer=tf.constant_initializer(0.04), trainable=False)
				self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
				self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-6))
				self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-6)*self.advantages)
				self.loss = self.value_loss + self.policy_loss - self.entropy * entropy_beta
				self.adv_sum = tf.reduce_sum(self.advantages)

				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
				
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
				self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

start_lr = 0.0001
lr_decay_ratio = 0.9
lr_decay_steps = 10000


class Worker():
	def __init__(self,name,s_size,a_size,model_path,global_episodes):
		self.name = "worker_" + str(name)
		self.number = name        
		self.model_path = model_path
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

		self.local_AC = AC_Network(s_size,a_size,self.name)
		self.update_local_ops = update_target_graph('global',self.name)        
		
		
		
		self.env = gameEnv(FLAGS.json_file,FLAGS.vocab_file)
	
	def getAttentionMaps(self,combinedMaps) :
		proc_maps = []
		_,x,y,z = combinedMaps.shape
		for i in range(z) :
			maps = combinedMaps[:,:,:,i]
			maps/= 255.0
			maps =  np.reshape(maps,[7,7])
			maps = np.reshape(scipy.misc.imresize(maps,[84,84]),[84,84,1])
			proc_maps.append(maps)
		return proc_maps	


	def work(self,max_episode_length,gamma,global_AC,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		lr = start_lr

		print("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():
			count = 0       
			          
			while not coord.should_stop():
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_frames1 = []
				attention_frames1 = []
				attention_frames2 = []
				attention_frames3 = []
				attention_frames4 = []
				attention_frames5 = []
				episode_reward = 0
				episode_step_count = 0
				
				s,reward_description,reward_sentence = self.env.reset()
				episode_frames1.append(s[1][...,[2,1,0]])
				s = s[0]
				episode_frames.append(s[...,[2,1,0]])

				s = process_frame(s)
				
				
				d = False
				rnn_state = self.local_AC.state_init
				rnn_out_state = sess.run(self.local_AC.reward_out,feed_dict = {self.local_AC.reward_desc:[reward_description],
					self.local_AC.state_in[0]:rnn_state[0],self.local_AC.state_in[1]:rnn_state[1]})
				
				while not d:
					a_dist,v,rnn_state,attention_map = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.attention_map],
						feed_dict={self.local_AC.inputs:[s],self.local_AC.reward_desc:[reward_description], 
						self.local_AC.state_in[0]:rnn_state[0],self.local_AC.state_in[1]:rnn_state[1]})
					
					attention_map = self.getAttentionMaps(attention_map)
					attention_frames1.append(attention_map[0])
					attention_frames2.append(attention_map[1])
					attention_frames3.append(attention_map[2])
					attention_frames4.append(attention_map[3])
					attention_frames5.append(attention_map[4])

					a = np.random.choice(a_dist[0],p=a_dist[0])
					a = np.argmax(a_dist == a)

					s1,r,d = self.env.step(a)
					episode_frames1.append(s1[1][...,[2,1,0]])
					s1 = s1[0]
					episode_frames.append(s1[...,[2,1,0]])

					s1 = process_frame(s1)
					

					episode_buffer.append([s,a,r,s1,d,v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s = s1                    
					total_steps += 1
					episode_step_count += 1
					
					if episode_step_count >= max_episode_length :
						break
				
				print('the episode length is %d and the episode reward is %g'%(len(episode_frames),episode_reward))   
				images = np.array(episode_frames)
				images1 = np.array(episode_frames1)
				make_gif(images,gif_path+str(reward_sentence.replace(' ','_'))+'image_'+str(count)+'.gif',
							duration=len(images)*0.5,true_image=True,salience=False)
				make_gif(images1,gif_path+str(reward_sentence.replace(' ','_'))+'Originalimage_'+str(count)+'.gif',
							duration=len(images1)*0.5,true_image=True,salience=False)
				make_gif(attention_frames1,gif_path+str(reward_sentence.replace(' ','_'))+'attention1_'+str(count)+'.gif',
							duration=len(attention_frames1)*0.5,true_image=True,salience=False)
				make_gif(attention_frames2,gif_path+str(reward_sentence.replace(' ','_'))+'attention2_'+str(count)+'.gif',
							duration=len(attention_frames2)*0.5,true_image=True,salience=False)
				make_gif(attention_frames3,gif_path+str(reward_sentence.replace(' ','_'))+'attention3_'+str(count)+'.gif',
							duration=len(attention_frames3)*0.5,true_image=True,salience=False)
				make_gif(attention_frames4,gif_path+str(reward_sentence.replace(' ','_'))+'attention4_'+str(count)+'.gif',
							duration=len(attention_frames4)*0.5,true_image=True,salience=False)
				make_gif(attention_frames5,gif_path+str(reward_sentence.replace(' ','_'))+'attention5_'+str(count)+'.gif',
							duration=len(attention_frames5)*0.5,true_image=True,salience=False)

				if count >= 0:
					break
				
				sess.run(self.increment)
				episode_count += 1
				count+=1

			
max_episode_length = FLAGS.max_episode_length
gamma = .99 
s_size = 84*84*3 
a_size = 4 
load_model = True
model_path = FLAGS.model_path
gif_path = FLAGS.gif_path


tf.reset_default_graph()

if not os.path.exists(model_path):
	os.makedirs(model_path)
	
if not os.path.exists(gif_path):
	os.makedirs(gif_path)


global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)


master_network = AC_Network(s_size,a_size,'global') 
num_workers = FLAGS.num_workers
workers = []

for i in range(num_workers):
	workers.append(Worker(i,s_size,a_size,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,inter_op_parallelism_threads=12)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	coord = tf.train.Coordinator()
	if load_model == True:
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
		print('Model loaded')
	else:
		sess.run(tf.global_variables_initializer())
		
	
	worker_threads = []
	for worker in workers:
		worker_work = lambda: worker.work(max_episode_length,gamma,master_network,sess,coord,saver)
		t = threading.Thread(target=(worker_work))
		t.start()
		sleep(0.5)
		worker_threads.append(t)
	coord.join(worker_threads)    
   
										  
