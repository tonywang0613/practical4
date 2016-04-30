import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import random

class util:
	def __init__(self):
		self.reward=0.5
		self.last_state=None
		self.last_action=None

		self.gamma=0.9

		
		self.model=Sequential()
		self.model.add(Dense(100,init='lecun_uniform',input_shape=(6,)))
		self.model.add(Activation('relu'))
		
		self.model.add(Dense(150,init='lecun_uniform'))
		self.model.add(Activation('relu'))
		
		self.model.add(Dense(2,init='lecun_uniform'))
		self.model.add(Activation('relu'))
		
		rms=RMSprop()
		self.model.compile(loss='mse',optimizer=rms)
		
		
		
	
	def action_callback(self,state):
		state=state
		last_state=self.last_state
		last_action=self.last_action
		model=self.model
		
		random_action=1 if random.random()>0.5 else 0
		
		#read the state information
		#create state matrix take four coordinate
		#dist_t: distance to the tree
		#height_t:the height to the tree: monkey_bot-tree_bot
		#vel:monkey_vel
		
		
		tdist,ttop,vbot,mvel,mtop,mbot=self.con_state(state)
		new_entry=np.array([tdist,ttop,vbot,mvel,mtop,mbot])
		#print "new entry{}".format(new_entry)
		
		
		
		
		
		
		
		if self.last_state==None:
			new_q=model.predict(new_entry.reshape(1,6),batch_size=1)
			new_action=1 if np.argmax(new_q)>np.argmax(new_q) else 0
			self.last_state=state
					
		elif not self.last_state==None:
			tdist_l,ttop_l,tbot_l,mvel_l,mtop_l,mbot_l=self.con_state(last_state)
			last_entry=np.array([tdist_l,ttop_l,tbot_l,mvel_l,mtop_l,mbot_l])
			
			new_q=model.predict(new_entry.reshape(1,6),batch_size=1)
			print new_q
			new_action=1 if new_q[0][0]>new_q[0][1] else 0
			y=np.zeros((1,2))
			y[:]=new_q[:]
			
			update=self.last_reward+self.gamma*np.max(new_q)
			#print update
			y[0][new_action]=update
			print y
			model.fit(new_entry.reshape(1,6),y,batch_size=1,nb_epoch=1,verbose=1)
			self.last_state=state
			self.model=model
			return new_action
			

			
			
			
	def reward_callback(self,reward):
		self.last_reward=reward
	
	#take state return four value, with normalized value
	def con_state(self,state):
		
		tdist=state['tree']['dist']
		ttop=state['tree']['top']
		tbot=state['tree']['bot']
		mvel=state['monkey']['vel']
		mtop=state['monkey']['top']
		mbot=state['monkey']['bot']
		
		return tdist,ttop,tbot,mvel,mtop,mbot

		
	