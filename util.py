import numpy as np
import random
class util:
	def __init__(self):
		self.reward=0.5
		self.last_state=None
		self.last_action=None
		self.bin=50
		self.EPS=0.01
		self.gamma=0.9
		self.Q=np.zeros((600/self.bin,400/self.bin,70,2))
		self.epsilon=np.zeros((600/self.bin,400/self.bin,70,2))
	
	def action_callback(self,state):
		state=state
		last_state=self.last_state
		last_action=self.last_action
		
		random_action=1 if random.random()>0.5 else 0
		new_action=random_action
		#read the state information
		#create state matrix take four coordinate
		#dist_t: distance to the tree
		#height_t:the height to the tree: monkey_bot-tree_bot
		#vel:monkey_vel
		
		dist,height,vel=self.con_state(state)
		
		if not self.last_state==None:
			dist_l,height_l,vel_l=self.con_state(last_state)
			
			max_q=max(self.getQ(state))
			print self.getQ(state)
			
			new_action=1 if self.getQ(state)[1]>self.getQ(state)[0] else 0
			
			if self.epsilon[dist,height,vel,new_action]>0:
				e=self.EPS/self.epsilon[dist,height,vel,new_action]
			else:
				e=self.EPS
			
			if (random.random()<e):
				new_action=random_action
			
			alpha=1/self.epsilon[dist_l,height_l,vel_l,last_action]
			self.Q[dist_l,height_l,vel_l,last_action]+=alpha*(self.last_reward+self.gamma*max_q-self.Q[dist_l,height_l,vel_l,last_action])
		self.last_action=new_action
		self.last_state=state
		self.epsilon[dist,height,vel,new_action]+=1
		print new_action
		return new_action
		
	
		
		print self.reward
		#return random_action
	
	def reward_callback(self,reward):
		self.last_reward=reward
	
	#take state return four value, with normalized value
	def con_state(self,state):
		bin=self.bin
		dist=state['tree']['dist']/bin
		height=(state['monkey']['bot']-state['tree']['bot'])/bin
		vel=state['monkey']['vel']+30
		return dist,height,vel
		
	def getQ(self,state):
		bin=self.bin
		dist=state['tree']['dist']/bin
		height=(state['monkey']['bot']-state['tree']['bot'])/bin
		vel=state['monkey']['vel']+30
		return self.Q[dist,height,vel]
		
	