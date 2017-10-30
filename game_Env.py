import numpy as np
from PIL import Image
import json
import random
import scipy.misc
import cv2
import generateSentence as gen_sen

class Env(object) :


	def __init__(self,json_file,vocab_file) :

		with open(json_file) as file :
			data = json.load(file)
		self.jsonData = data
		
		with open(vocab_file) as f:
			vocabulary = f.readlines()
		vocabulary = [x.strip("\n") for x in vocabulary]
		self.vocabulary = vocabulary

		env_info = data["environment"]
		self.image_w = env_info["Width"]
		self.image_h = env_info["Height"]
		self.grid_w = env_info["Block_width"]
		self.grid_h = env_info["Block_height"]
		self.step_w = int(self.image_w/self.grid_w)
		self.step_h = int(self.image_h/self.grid_h)
		self.sizes = env_info["Sizes"]
		self.game_w = self.step_w*self.grid_w
		self.game_h = self.step_h*self.grid_h
		self.fpRadius = 3
		
		self.background_color = env_info["Background"]
		self.initialise()
		

		agent_info = data["agent"][0]
		self.agent_size = self.sizes[agent_info["size"]]
		#self.agent_image = np.array(Image.open(agent_info["img"]))
		self.agent_image = cv2.imread(agent_info["img"],1)
		rewards_info = data["objects"]
		self.objects = {}
		self.rewardIDS = []
		for reward in rewards_info :
			self.addObject(reward,-0.5,False)	
		
		blockages_info = data["obstacles"]
		
		for blockage in blockages_info :
			self.addObject(blockage,blockage["reward"],True)	
			

		# print self.blockage_sizes

	def addObject(self,objInfo,reward,isObstacle) :
		obj = []
		obj.append(self.sizes[objInfo["size"]])
		obj.append(cv2.imread(objInfo["img"],1))
		obj.append(reward)
		obj.append(objInfo["id"])
		obj.append(objInfo["hard"])
		obj.append(isObstacle)
		self.objects[objInfo["id"]] = obj
		if isObstacle == False :
			self.rewardIDS.append(objInfo["id"])

	def initialise(self) :
		self.game = np.zeros([self.game_h,self.game_w,3])
		self.game_map = np.zeros([self.grid_h,self.grid_w])
		self.reward_map = np.zeros([self.grid_h,self.grid_w])
		self.ids = np.zeros([self.grid_h,self.grid_w])
		self.x_position = np.zeros([self.grid_h,self.grid_w])
		self.y_position = np.zeros([self.grid_h,self.grid_w])
		self.reward_dict = []
		self.game = self.update_color(self.game,self.background_color)
		self.static_pos = (-1,-1)

	def placeBlocks(self) :
		self.initialise()

		self.agent_x,self.agent_y = self.add_block((self.agent_size,self.agent_size),self.agent_image,-2,99,1)
		currentIds = np.random.choice(self.rewardIDS,np.random.randint(3,7),replace = False)
		for idx in self.objects :
			obj = self.objects[idx]
			if obj[5] == True or(idx in currentIds) :
				map_val = obj[4]
				if obj[5] == False :
					if obj[4] == 1 :
						map_val = 2
					else :
						map_val = 0	

				pos = self.add_block((obj[0],obj[0]),obj[1],obj[2],obj[3],map_val)
				if obj[5] == False :
					self.reward_dict.append([idx,pos])
		return currentIds	
				
	def makeStaticReward(self,pos) :
		self.static_pos = pos

	def ensureDistanceReward(self,sentence,pos) :
		if 'north' in sentence :
			self.makeStaticReward((pos[0]+1,pos[1]))
		elif 'south' in sentence :
			self.makeStaticReward((pos[0]-1,pos[1]))
		elif 'east' in sentence :
			self.makeStaticReward((pos[0],pos[1]-1))
		else :
			self.makeStaticReward((pos[0],pos[1]+1))			
	def reset(self) :
		currentIds = self.placeBlocks()
		rewardInfo = gen_sen.bfs(self.game_map,(self.agent_x,self.agent_y),self.reward_dict, self.jsonData, self.vocabulary,self.ids, currentIds)
		
		while len(rewardInfo) == 0 :
			currentIds = self.placeBlocks()
			rewardInfo = gen_sen.bfs(self.game_map,(self.agent_x,self.agent_y),self.reward_dict, self.jsonData, self.vocabulary,self.ids, currentIds)

		for reward in rewardInfo[2] :
			self.reward_map[reward[0][0]:reward[0][0]+reward[1],reward[0][1]:reward[0][1]+reward[1]] = 1
			if rewardInfo[3] :
				self.ensureDistanceReward(rewardInfo[0],(reward[0][0],reward[0][1]))	
		
		return self.get_state(),np.array(rewardInfo[1]),rewardInfo[0]

			
	def collisionWithAnyBlock(self,pos,dims) :
		if np.max(self.ids[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]) > 0 :
			return True
		if np.max(self.reward_map[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]) == 1:
			return True	
		return False

	def getPosition(self,size_x,size_y) :
		pos_x = random.randint(0,self.grid_h- size_y-1)
		pos_y = random.randint(0,self.grid_w- size_x-1)
		
		while (self.collisionWithAnyBlock((pos_x,pos_y),(size_x,size_y))) :
			pos_x = random.randint(0,self.grid_h- size_y-1)
			pos_y = random.randint(0,self.grid_w- size_x-1)

		return (pos_x,pos_y)	
	
	def get_fake_fpstate(self, image):
		mask = np.zeros([self.image_h, self.image_w, 3])
		st_x = (self.image_h - self.game_h)/2
		st_y = (self.image_w - self.game_w)/2
		#egoView = np.zeros([self.image_h, self.image_w, 3])
		if st_x+(self.agent_x - self.fpRadius)*self.step_h < 0 and st_y+(self.agent_y - self.fpRadius)*self.step_w < 0:
			mask[0:st_x+(self.agent_x+self.fpRadius)*self.step_h, 0:st_y+(self.agent_y+self.fpRadius)*self.step_w, :] = 1.0
		elif st_x+(self.agent_x - self.fpRadius)*self.step_h < 0:
			mask[0:st_x+(self.agent_x+self.fpRadius)*self.step_h, st_y+(self.agent_y-self.fpRadius)*self.step_w:st_y+(self.agent_y+self.fpRadius)*self.step_w, :] = 1.0
		elif st_y+(self.agent_y - self.fpRadius)*self.step_w < 0:
			mask[st_x+(self.agent_x-self.fpRadius)*self.step_h:st_x+(self.agent_x+self.fpRadius)*self.step_h, 0:st_y+(self.agent_y+self.fpRadius)*self.step_w, :] = 1.0
		else:
			mask[st_x+(self.agent_x-self.fpRadius)*self.step_h:st_x+(self.agent_x+self.fpRadius)*self.step_h, st_y+(self.agent_y-self.fpRadius)*self.step_w:st_y+(self.agent_y+self.fpRadius)*self.step_w, :] = 1.0
		
		fpView = np.multiply(image, mask)

		return [fpView, image]

	def get_actual_fpstate(self, image):
		egoView = np.zeros([self.image_h, self.image_w, 3])
		dia_x = 2*self.step_h*self.fpRadius
		dia_y = 2*self.step_w*self.fpRadius
		mod_h = self.image_h + dia_x
		mod_w = self.image_w + dia_y
		modView = np.zeros([mod_h, mod_w, 3])
		modView[(self.fpRadius*self.step_h):(self.fpRadius*self.step_h+self.image_h), (self.fpRadius*self.step_w):(self.fpRadius*self.step_w+self.image_w), :] = image
		start_x = (self.image_h - dia_x)/2
		start_y = (self.image_w - dia_y)/2
		st_x = (self.image_h - self.game_h)/2
		st_y = (self.image_w - self.game_w)/2
		egoView[start_x:start_x+dia_x, start_y:start_y+dia_y, :] = \
		modView[(st_x+(self.agent_x+self.fpRadius-self.fpRadius)*self.step_h):(st_x+(self.agent_x+self.fpRadius+self.fpRadius)*self.step_h), (st_y+(self.agent_y+self.fpRadius-self.fpRadius)*self.step_w):(st_y+(self.agent_y+self.fpRadius+self.fpRadius)*self.step_w), :]

		return [egoView, image]

	def get_state(self) :
		image = np.zeros([self.image_h,self.image_w,3])
		start_x = (self.image_h - self.game_h)/2
		start_y = (self.image_w - self.game_w)/2

		image[start_x:start_x+self.game_h,start_y:start_y+self.game_w,:] = self.game
		return self.get_actual_fpstate(image)
	
	def add_pos_block(self,pos,size,img,reward,ID,map_val,resize = True) :
			
		# self.game[start_x:start_x+self.step_x-1,start_y:start_y+self.step_y-1,:] = board
		self.game_map[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = map_val
		if resize :
			img = scipy.misc.imresize(img,(size[0]*self.step_h,size[1]*self.step_w))
		
		self.game[pos[0]*self.step_h:(pos[0]+size[0])*self.step_h,pos[1]*self.step_w:(pos[1]+size[1])*self.step_w,:] = img
		self.reward_map[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = reward
		self.ids[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = ID
		self.x_position[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = pos[0]
		self.y_position[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = pos[1]
	
	def add_block(self,size,img,reward,ID,map_val) :
		pos_x,pos_y = self.getPosition(size[0],size[1])
		self.add_pos_block((pos_x,pos_y),size,img,reward,ID,map_val)
		return (pos_x,pos_y)

	
	def remove_block(self,pos,size) :
		pos = (int(pos[0]),int(pos[1]))
		size = (int(size[0]),int(size[1]))
		img = np.zeros([size[0]*self.step_h,size[1]*self.step_w,3])
		img1 = self.update_color(img,self.background_color)
		
		self.add_pos_block(pos,size,img1,0,0,0,resize = False)
		self.game_map[pos[0]:pos[0]+size[0],pos[1]:pos[1]+size[1]] = 0

	def update_color(self,board,color) :
		for i in range(3) :
			board[:,:,i] = color[i]
		return board		

	def step(self,action) :
		'''
		0- down
		1-top
		2-left
		3-right
		'''
		done = False
		reward = 0.0
		
		
		
		new_pos = (-1,-1)
		
		if action == 0 and self.agent_x < self.grid_h-1 :
			new_pos = (self.agent_x+1,self.agent_y)
		
		elif action == 1 and self.agent_x > 0 :
			new_pos = (self.agent_x-1,self.agent_y)

		elif action == 2 and self.agent_y > 0 :
			new_pos = (self.agent_x,self.agent_y-1)

		elif action == 3 and self.agent_y < self.grid_w - 1 :
			new_pos = (self.agent_x,self.agent_y+1)

		#print self.reward_map	
		if not new_pos == (-1,-1) :
			reward = self.reward_map[new_pos[0],new_pos[1]]
			ID = int(self.ids[new_pos[0],new_pos[1]])
			if reward == 1 :
				done = True
			self.remove_block((self.agent_x,self.agent_y),(self.agent_size,self.agent_size))
			if ID == 0 :
				self.add_pos_block(new_pos,(self.agent_size,self.agent_size),self.agent_image,-2,99,1)
				self.agent_x = new_pos[0]
				self.agent_y = new_pos[1]
			
			elif ((self.objects[ID][4] == 0 and (not self.static_pos == new_pos)) or done == True) :

				
				obj = self.objects[ID]
				self.remove_block((self.x_position[new_pos[0],new_pos[1]],self.y_position[new_pos[0],new_pos[1]]),(obj[0],obj[0]))
				self.add_pos_block(new_pos,(self.agent_size,self.agent_size),self.agent_image,-2,99,1)
				if not done :
					self.add_block((obj[0],obj[0]),obj[1],obj[2],obj[3],obj[4])
				self.agent_x = new_pos[0]
				self.agent_y = new_pos[1]

			elif self.objects[ID][4] == 1 or self.static_pos == new_pos:
				
				self.add_pos_block((self.agent_x,self.agent_y),(self.agent_size,self.agent_size),self.agent_image,-2,99,1)

		next_state = self.get_state()

		return next_state,reward,done	
if __name__ == '__main__' :
	
	env = Env("env_specs1.json")
	x = env.reset()
