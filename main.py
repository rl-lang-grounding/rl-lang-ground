import threading
import multiprocessing
import numpy as np
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
flags.DEFINE_integer("total_attention_maps", 5, "number of attention maps in model")
flags.DEFINE_string("model_path",'./model/',"the directory where the models are going to be saved")
flags.DEFINE_string("gif_path",'./languageFrames/',"the directory where the gifs are going to be stored")
flags.DEFINE_string("json_file",'objects.json',"the json file descibing the environment")
flags.DEFINE_string("vocab_file",'vocab.txt',"the file containing vocabulary words")
flags.DEFINE_integer("max_episode_length", 100, "maximum length of episode")
flags.DEFINE_integer("num_workers", 32, "number of threads")

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
					
					for i in range(FLAGS.total_attention_maps) :
						reward_out = outputs[-1]
						reward_out = slim.fully_connected(reward_out,64,activation_fn=None)
						reward_out = PReLU(reward_out)
						reward_out = tf.reshape(reward_out, [1,1,64,1])
						attention = tf.nn.conv2d(net, reward_out, strides=[1, 1, 1, 1], padding='VALID')
						if i == 0 :
							attention_maps = attention
						else :
							attention_maps = tf.concat([attention_maps,attention],3)

					hidden = attention_maps		

					
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
		
	def train(self,global_AC,rollout,sess,gamma,bootstrap_value,reward_description, learning_rate):
		rollout = np.array(rollout)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]	
		
		
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs:np.vstack(observations),
			self.local_AC.reward_desc:[reward_description],
			self.local_AC.actions:actions,
			self.local_AC.advantages:advantages,
			self.local_AC.lr : learning_rate,
			self.local_AC.state_in[0]:self.batch_rnn_state[0],
			self.local_AC.state_in[1]:self.batch_rnn_state[1]
			}
		v_l,p_l,e_l,g_n,v_n,adv, apl_g, self.batch_rnn_state = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.adv_sum,
			self.local_AC.apply_grads,
			self.local_AC.state_out],
			feed_dict=feed_dict)
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n, v_n, adv/len(rollout)
		
	def work(self,max_episode_length,gamma,global_AC,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		lr = start_lr

		print("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():                 
			while not coord.should_stop():
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				
				s,reward_description,reward_sentence = self.env.reset()
				episode_frames.append(s[1])
				s = s[0]
				s = process_frame(s)
				rnn_state = self.local_AC.state_init
				self.batch_rnn_state = rnn_state

				
				
				d = False
				if episode_count % lr_decay_steps == 0 and not episode_count == 0 :
					lr *= lr_decay_ratio
				while not d:
					
					a_dist,v, rnn_state = sess.run([self.local_AC.policy,self.local_AC.value, self.local_AC.state_out], 
						feed_dict={self.local_AC.inputs:[s],self.local_AC.reward_desc:[reward_description],
						self.local_AC.state_in[0]:rnn_state[0],
						self.local_AC.state_in[1]:rnn_state[1]})

					a = np.random.choice(a_dist[0],p=a_dist[0])
					a = np.argmax(a_dist == a)

					s1,r,d = self.env.step(a)

					episode_frames.append(s1[1])
					s1 = s1[0]
					s1 = process_frame(s1)
					
					episode_buffer.append([s,a,r,s1,d,v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s = s1
								   
					total_steps += 1
					episode_step_count += 1

					
					if len(episode_buffer) == 20 and d != True:
						
						v1 = sess.run(self.local_AC.value, 
							feed_dict={self.local_AC.inputs:[s],self.local_AC.reward_desc:[reward_description],
							self.local_AC.state_in[0]:rnn_state[0],
							self.local_AC.state_in[1]:rnn_state[1]})[0,0]
						v_l,p_l,e_l,g_n,v_n, adv = self.train(global_AC,episode_buffer,sess,gamma,v1,reward_description, lr)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if  episode_step_count >= max_episode_length - 1 or d == True:
						break

				if episode_count %lr_decay_steps == 0 :
					print('the learning rate is %g'%lr)		
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				
				
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n,adv = self.train(global_AC,episode_buffer,sess,gamma,0.0,reward_description, lr)
								
					
				
				if episode_count % 5 == 0 and episode_count != 0:
					if episode_count % 1000 == 0 and self.name == 'worker_0':
						saver.save(sess,model_path+ str(episode_count)+'.ckpt')
						print("Saved Model")
					if self.name == 'worker_0' and episode_count % 250 == 0:
						time_per_step = 0.2
						if len(episode_frames) > 100 :
							images = np.array(episode_frames)[-100:]
						else :
							images = np.array(episode_frames)	
						make_gif(images,gif_path+str(reward_sentence.replace(' ','_'))+'image_'+str(episode_count)+'.gif',
							duration=len(images)*time_per_step,true_image=True,salience=False)


					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_length = np.mean(self.episode_lengths[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])
					if episode_count%50 == 0 and self.name == 'worker_0' :
						print('%d steps reached and the mean reward is %g and mean length is %g'%(episode_count,float(mean_reward),float(mean_length)));sys.stdout.flush()
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Advantage', simple_value=float(adv))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				
				sess.run(self.increment)
				episode_count += 1

max_episode_length = FLAGS.max_episode_length
gamma = .99 
s_size = 84*84*3 
a_size = 4
load_model = False
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
config=tf.ConfigProto(log_device_placement=False,inter_op_parallelism_threads=12)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	coord = tf.train.Coordinator()
	if load_model == True:
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
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
   
										  
