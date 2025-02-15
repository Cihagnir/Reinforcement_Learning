
from collections import deque 
from random import sample


class Replay_Memory(object) : 

  def __init__(self, max_size, batch_size) : 

    self.memory = deque(maxlen = max_size)
    self.batch_size = batch_size

  def push(self, state, action, reward, next_state, is_over) : 
    '''
    Push function is used to push the data into the memory. 

    @param state : The current state of the game as tensor 
    @param action : The action taken by the agent as tensor 
    @param reward : The reward received by the agent as tensor 
    @param next_state : The next state of the game as tensor 
    @param done : The flag to check if the game is over or not as tensor  
    '''

    self.memory.append((state, action, reward, next_state, is_over))

  def sample(self) : 
    '''
    sample function is used to sample the data from the memory. 
    '''

    return sample(self.memory, self.batch_size)


class Avrg_Reward_Que(object) : 

  def __init__(self, max_size)  :

    self.memory = deque(maxlen= max_size)
    self.past_avrg_reward = float('-inf')

  def push(self, episode_reward) : 
    self.memory.append(episode_reward)

  def avrg_reward(self) : 
    
    avrg = float('-inf')
    if len(self.memory ) != 0 :
      avrg = sum(self.memory) / len(self.memory)
    
    return avrg

  def is_model_better(self) : 

    is_better = False
    average = self.avrg_reward()
    if  average > self.past_avrg_reward :
      is_better = True
      self.past_avrg_reward = average

    return is_better