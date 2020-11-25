###import dependencies
import random
import gym
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation ,Dense
from tensorflow.keras.optimizers import Adam
import os

#set parameters
env =gym.make('CartPole-v1')

state_size =env.observation_space.shape[0] #cart_pole ko state haru
state_size

action_size =env.action_space.n #action haru
action_size

batch_size=32

n_episodes=1001

output_dir = 'model_output/cartpole'# to store our model output
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

#define agent
class DQNAgent:
  def __init__(self,state_size,action_size):
    self.state_size=state_size
    self.action_size=action_size

    self.memory=deque(maxlen=2000) #random sample haru yesma  basxaa epiosde ko
    
    self.gamma =0.95 #discount factor

    self.epsilon=1.0    #exploration_explotation
    self.epsilon_decay=0.995 #decay exploration over time
    self.epsilon_min=0.01    #min amount of exploration 
    
    self.learning_rate=0.001

    
    self.model=self._build_model()
  
  def _build_model(self):
    model= Sequential()
    model.add(Dense(24,input_dim=self.state_size,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(self.action_size,activation='linear'))
    model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
    
    return model
  
  #at current time step #done le episode end or not
  def remember(self,state,action,reward,next_state,done):
    self.memory.append((state,action,reward,next_state,done))
  
  def act(self,state):
    if np.random.rand() <=self.epsilon:
      return random.randrange(self.action_size)
    
    act_values=self.model.predict(state) #exploite
    return np.argmax(act_values[0])
  
  def replay(self,batch_size):
    minibatch=random.sample(self.memory,batch_size)

    for state,action,reward,next_state,done in minibatch:
      target= reward
      if not done: #estimate future reward
        target= (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

      #now max mapp maximize future reward to current reward
      target_f=self.model.predict(state)
      target_f[0][action]=target

      self.model.fit(state,target_f,epochs=1,verbose=0)
    
    if self.epsilon > self.epsilon_min:
      self.epsilon *=self.epsilon_decay

  def load(self,name):
    self.model.load_weights(name)
      
  def save(self,name):
    self.model.save_weights(name)    

agent= DQNAgent(state_size,action_size)

#INTERACTING WITH AN OPENAI GYM ENVIRONMENT
done=False
for e in range(n_episodes):
  
  state=env.reset()
  state=np.reshape(state,[1,state_size]) #row to column for SNN
#itrn over timesteps that our episode can run for
  for time in range(5000):
    env.render()
    action=agent.act(state)
     
    next_state,reward,done, _ = env.step(action)

    reward=reward if not done else -10

    next_state=np.reshape(next_state,[1,state_size])

    agent.remember(state,action,reward,next_state,done)

    state =next_state


    if done:
      print('episode:{}/{},score:{},e:{:.2}'.format(e,n_episodes,time,agent.epsilon))
      break
 #upate theta weights
  if len (agent.memory) > batch_size:
    agent.replay(batch_size)
  
  if e % 50 ==0:
    agent.save(output_dir + 'weights_' +'{:04d}'.format(e) + '.hdf5')



