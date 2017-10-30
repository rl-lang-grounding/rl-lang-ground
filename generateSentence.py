import numpy as np
from collections import deque
import json
import itertools
import random


def getReachablePoints(pos,size) :
	x,y = size
	if pos[0] == 0 :
		if pos[1] == 0 :
			return [1,1*x+0]
		elif pos[1] == y-1 :
			return [pos[1]-1,1*x+pos[1]]
		else :
			return [pos[1]-1,pos[1]+1,1*x+pos[1]]
	elif pos[1] == y-1 :
		if pos[0] == x-1 :
			return [pos[0]*x+y-2,(pos[0]-1)*x+y-1]
		else :
			return [pos[0]*x+y-2,(pos[0]-1)*x+y-1,(pos[0]+1)*x+y-1]
	elif pos[0] == x-1 :
		if pos[1] == 0 :
			return [(pos[0]-1)*x+pos[1],pos[0]*x+(pos[1]+1)]
		else :
			return [(pos[0]-1)*x+pos[1],pos[0]*x+(pos[1]+1),pos[0]*x+(pos[1]-1)]
	elif pos[1] == 0 :
		return [pos[0]*x+1,(pos[0]-1)*x,(pos[0]+1)*x]
	else :
		return [(pos[0]-1)*x+pos[1],(pos[0]+1)*x+pos[1],pos[0]*x+pos[1]-1,pos[0]*x+pos[1]+1]

def addNodes(pos,size,flags,q,game_map) :
	nodes = getReachablePoints(pos,size)
	for node in nodes :
		if flags[node] == 0 and game_map[node/size[0],node%size[0]] == 0:
			flags[node] = 1
			q.append(node)
		if game_map[node/size[0],node%size[0]] == 2 :
			flags[node] = 1 	
	return q,flags

def convertOneHot(idx,l) :
	arr = np.zeros([l])
	arr[idx] = 1
	return arr

def checkVocab(s, vocabulary):
	fl = 0
	for i in s.split(" "):
		if i not in vocabulary:
			fl = 1
	return fl

# Case 4: Go to <object type>

def genObjectSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, active_rewards):
	ids1 = []
	for item in reward_dict :
		ids1.append(item[0])

	ids1 = np.array(ids1)	

	for item in data["objects"]:
		
		ide = item["id"]
		if not ide in active_rewards:
			continue

		req_ind = np.where(ids1 == ide)	
		req_ind = req_ind[0][0]

		loc = reward_dict[req_ind][1]
		sz = data["environment"]["Sizes"][item["size"]]

		if reachable[loc] == 1.0:
			s = "Go to " + item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					pos_size[ind].append((loc, sz))
				else:
					possible_sentences.append(s)
					pos_size.append([(loc, sz)])


	return possible_sentences, pos_size

#Case 7: Go to <location> of <type>
def genLocSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, ids, active_rewards):
	ids1 = []
	for item in reward_dict :
		ids1.append(item[0])
	ids1 = np.array(ids1)	

	rows, cols = reachable.shape
	for item in data["objects"]:
		
		if item["size"] != "small":
			continue
		
		ide = item["id"]

		if not ide in active_rewards:
			continue

		req_ind = np.where(ids1 == ide)
		req_ind = req_ind[0][0]

		rx, ry = reward_dict[req_ind][1]
		
		size = data["environment"]["Sizes"][item["size"]]
		
		flagn = 0
		flags = 0
		flage = 0
		flagw = 0
		pos_n = []
		pos_s = []
		pos_e = []
		pos_w = []		

		for s in range(size):
			if ry+s <= cols-1 and rx-1 >= 0:
				if reachable[rx-1][ry+s] == 1.0 and ids[rx-1][ry+s] == 0.0:
					pos_n.append(((rx-1, ry+s), 1))
					flagn = 1

			if ry+s <= cols-1 and rx+size <= rows-1:
				if reachable[rx+size][ry+s] == 1.0 and ids[rx+size][ry+s] == 0.0:
					pos_s.append(((rx+size, ry+s), 1))
					flags = 1

			if ry-1 >= 0 and rx+s <= rows-1:
				if reachable[rx+s][ry-1] == 1.0 and ids[rx+s][ry-1] == 0.0: 
					pos_w.append(((rx+s, ry-1), 1))
					flagw = 1

			if ry+size <= cols-1 and rx+s <= rows-1:
				if reachable[rx+s][ry+size] == 1.0 and ids[rx+s][ry+size] == 0.0:
					pos_e.append(((rx+s, ry+size), 1))
					flage = 1

		if flagn == 1:
			s = "Go to north of " + item["size"]+ " " + item["color"] + " " +item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					for ele in pos_n:
						pos_size[ind].append(ele)
				else:
					possible_sentences.append(s)
					pos_size.append(pos_n)

		if flags == 1:
			s = "Go to south of " + item["size"]+ " " +item["color"] + " " + item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					for ele in pos_s:
						pos_size[ind].append(ele)
				else:
					possible_sentences.append(s)
					pos_size.append(pos_s)

		if flagw == 1:
			s = "Go to west of " + item["size"]+ " " +item["color"] + " " + item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					for ele in pos_w:
						pos_size[ind].append(ele)
				else:
					possible_sentences.append(s)
					pos_size.append(pos_w)

		if flage == 1:
			s = "Go to east of " + item["size"]+ " " +item["color"] + " " + item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					for ele in pos_e:
						pos_size[ind].append(ele)
				else:
					possible_sentences.append(s)
					pos_size.append(pos_e)
	
	return possible_sentences, pos_size

# Case 8: Go to top/bottom right/left corners
def genCornerSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, ids):
	rows, cols = reachable.shape
	if reachable[0][0] == 1.0 and ids[0][0] == 0.0:
		s = "Go to top left corner"
		fl = checkVocab(s, vocabulary)
		if fl == 0:
			possible_sentences.append(s)
			pos_size.append([((0,0), 1)])

	if reachable[0][cols-1] == 1.0 and ids[0][cols-1] == 0.0:
		s = "Go to top right corner"
		fl = checkVocab(s, vocabulary)
		if fl == 0:
			possible_sentences.append(s)
			pos_size.append([((0,cols-1), 1)])

	if reachable[rows-1][0] == 1.0 and ids[rows-1][0] == 0.0:
		s = "Go to bottom left corner"
		fl = checkVocab(s, vocabulary)
		if fl == 0:
			possible_sentences.append(s)
			pos_size.append([((rows-1,0), 1)])

	if reachable[rows-1][cols-1] == 1.0 and ids[rows-1][cols-1] == 0.0:
		s = "Go to bottom right corner"
		fl = checkVocab(s, vocabulary)
		if fl == 0:
			possible_sentences.append(s)
			pos_size.append([((rows-1,cols-1), 1)])


	return possible_sentences, pos_size

# Case 9: Go to <color> <object type>
# Case 10: Go to <size> <object type>
def genAttrSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, active_rewards):
	
	obj_name = []
	obj_attr = []
	obj_locsz = []

	ids1 = []
	reachable_objs = []
	reachable_objs_attr = []

	for item in reward_dict :
		ids1.append(item[0])
	ids1 = np.array(ids1)	

	for item in data["objects"]:
		
		ide = item["id"]
		if not ide in active_rewards:
			continue

		req_ind = np.where(ids1 == ide)	
		req_ind = req_ind[0][0]
		#print req_ind
		loc = reward_dict[req_ind][1]
		sz = data["environment"]["Sizes"][item["size"]]
		
		if reachable[loc] == 1.0:
			name = item["size"] + " " + item["color"] +" "+item["type"]
			if not name in reachable_objs :
				reachable_objs.append(name)
				reachable_objs_attr.append([(loc,sz)])

			name = item["color"] +" "+item["type"]		
			if name in obj_name:
				ind = obj_name.index(name)
				obj_attr[ind].append((item["size"], item["color"]))
				obj_locsz[ind].append((loc, sz))
			else:
				obj_name.append(name)
				obj_attr.append([(item["size"], item["color"])])
				obj_locsz.append([(loc, sz)])

			s = "Go to " + item["color"] + " " + item["type"]
			fl = checkVocab(s, vocabulary)
			
			if fl == 0:
				if s in possible_sentences:
					ind = possible_sentences.index(s)
					pos_size[ind].append((loc, sz))
				else:
					possible_sentences.append(s)
					pos_size.append([(loc, sz)])

			ts = "There is a " + item["color"] + " " + item["type"] + " Go to it"
			fl = checkVocab(ts, vocabulary)
			
			if fl == 0:
				if ts in possible_sentences:
					ind = possible_sentences.index(ts)
					pos_size[ind].append((loc, sz))
				else:
					possible_sentences.append(ts)
					pos_size.append([(loc, sz)])


			

			s1 = "Go to " + item["size"] + " " + item["color"] + " " + item["type"]
			fl = checkVocab(s1, vocabulary)
			
			if fl == 0:
				if s1 in possible_sentences:
					ind = possible_sentences.index(s1)
					pos_size[ind].append((loc, sz))
				else:
					possible_sentences.append(s1)
					pos_size.append([(loc, sz)])
			
			ts1 = "There is a " + item["size"] + " " + item["color"] + " " + item["type"] + " Go to it"
			fl = checkVocab(ts1, vocabulary)
			
			if fl == 0:
				if ts1 in possible_sentences:
					ind = possible_sentences.index(ts1)
					pos_size[ind].append((loc, sz))
				else:
					possible_sentences.append(ts1)
					pos_size.append([(loc, sz)])

			
		
	for i in range(len(obj_name)):
		if len(obj_attr[i]) > 1:
			
			tsd2 = "There are multiple " + obj_name[i] + " Go to larger one"
			tsd3 = "There are multiple " + obj_name[i] + " Go to smaller one"			
			req_small = req_large = prev_large = 1
			prev_small = 3
			
			for j in range(len(obj_locsz[i])):
				if obj_locsz[i][j][1] < prev_small:
					prev_small = obj_locsz[i][j][1]
					req_small = obj_locsz[i][j]
				if obj_locsz[i][j][1] > prev_large:
					prev_large = obj_locsz[i][j][1]
					req_large = obj_locsz[i][j]
			
			fl2 = checkVocab(tsd2, vocabulary)
			if fl2 == 0:
				if req_large != 1 and req_small != 1:
					possible_sentences.append(tsd2)
					pos_size.append([req_large])
			fl3 = checkVocab(tsd3, vocabulary)
			if fl3 == 0:
				if req_small != 1 and req_large != 1:
					possible_sentences.append(tsd3)
					pos_size.append([req_small])
	
							
	return possible_sentences, pos_size

def isDistanceinSentence(sentence) :
	if 'north' in sentence :
		return True
	if 'south' in sentence :
		return True
	if 'east' in sentence :
		return True
	if 'west' in sentence :
		return True
	return False

def genPossibleSentences(reachable, reward_dict, data, vocabulary,ids, active_rewards):
	
	
	max_sentence_length = 9

	possible_sentences = []
	pos_size = []
	
	obj_name = []
	obj_attr = []
	obj_locsz = [] 

		
	possible_sentences, pos_size = genObjectSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, active_rewards)	
			
	possible_sentences, pos_size = genLocSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, ids, active_rewards)

	possible_sentences, pos_size = genCornerSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, ids)
	
	possible_sentences, pos_size = genAttrSentences(reachable, reward_dict, data, possible_sentences, pos_size, vocabulary, active_rewards)

	
	for s in possible_sentences:
		print(s)

	
	if len(possible_sentences) == 0 :
		return []
	index_value = random.sample(list(enumerate(possible_sentences)), 1)
	s = index_value[0][1]
	distanceInSentence = isDistanceinSentence(s)
	s = s.split(" ")
	rem = max_sentence_length - len(s)
	char_to_int = dict((c, i) for i, c in enumerate(vocabulary))
	int_to_char = dict((i, c) for i, c in enumerate(vocabulary))
	integer_encoded = [char_to_int[char] for char in s]
	onehot_encoded = list()

	for value in integer_encoded:
		letter = [0 for _ in range(len(vocabulary))]
		letter[value] = 1
		onehot_encoded.append(letter)

	pad = [0]*len(vocabulary)
	for r in range(rem):
		onehot_encoded.append(pad)

	
	return [index_value[0][1], onehot_encoded, pos_size[index_value[0][0]], distanceInSentence]



def bfs(game_map,start_pos,reward_dict, data, vocabulary,ids, active_rewards) :
	x,y = game_map.shape
	flags = np.zeros([x*y])
	flags[start_pos[0]*x+start_pos[1]] = 1
	q = deque([])
	q.append(start_pos[0]*x+start_pos[1])
	while q :
		pos = q.popleft()
		q,flags = addNodes((pos/x,pos%x),(x,y),flags,q,game_map)

	
	reachable = np.zeros([x,y])
	for i in range(x) :
		for j in range(y) :
			reachable[i,j] = flags[i*x+j]			
	arr = genPossibleSentences(reachable, reward_dict, data, vocabulary,ids, active_rewards)
	
	return arr
	

# if __name__ == '__main__' :
# 	a = np.zeros([3,3])
# 	start_pos = (1,1)
# 	a[1,0] = 1	
# 	a[2,1] = 1
# 	reachable, reward_dict = bfs(a,start_pos,[[0,(2,2)],[1,(0,2)],[2,(2,0)]])
# 	print('reachable is ', reachable)
# 	print('reward dict is ', reward_dict)

