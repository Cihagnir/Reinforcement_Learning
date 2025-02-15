# Import Section 
import torch 
from torch import nn 
from torch.optim import Adam
from torch.distributions import Normal
from torch.nn.functional import relu

class Q_Network(torch.nn.Module):

  def __init__(self, state_dim, action_dim, ):
    super(Q_Network, self).__init__()

    self.input_layer = nn.Linear(state_dim + action_dim, 256)
    self.hidden_layer_one = nn.Linear(256, 256)
    self.output_layer = nn.Linear(256, 1)


  def forward(self, state, action) :
    '''
    @param x : Tensor indclude the Env state information & Action taken by policy 
    @return output : Q-Value prediction if the action taken. 
    Dimension of the output layer could change between 'one' to 'number of  action' depending on your desing chose.   
    '''
    x = torch.cat((state, action), dim=-1)


    x = self.input_layer(x)
    x = relu(self.hidden_layer_one(x))
    output = self.output_layer(x)
    return output


class Policy_Network(torch.nn.Module) : 

  def __init__(self, state_dim, action_dim):
    super(Policy_Network, self).__init__()

    self.input_layer = nn.Linear(state_dim, 256)
    self.hidden_layer_one = nn.Linear(256, 256)
    self.output_layer_mean = nn.Linear(256, action_dim)
    self.output_layer_standart_dev = nn.Linear(256, action_dim)

  def forward(self, x) :
    '''
    @param x : Env state information 
    @return output_one : Mean info for Gausssion Distrib
    @return output_two : Standart dev info for Gausssion Distrib
    '''
    STD_UPPER = 10
    STD_LOWER = -10
    x = relu(self.input_layer(x))
    x = relu(self.hidden_layer_one(x))
    means = self.output_layer_mean(x)
    log_std_prev = self.output_layer_standart_dev(x)
    log_std = torch.clamp(log_std_prev, STD_LOWER, STD_UPPER)
    
    return  means, log_std


class Agent() : 

  def __init__(self, state_dim, action_dim, alpha, epsilon, gamma, q_net_lr, policy_lr, device, polyak = None):
    super().__init__()

    # Define the constatn used on the calculation
    self.alpha = alpha
    self.gamma = gamma
    self.polyak = polyak
    self.epsilon = epsilon

    # Define the networks 
    self.q_net_one = Q_Network(state_dim, action_dim).to(device)
    self.q_net_two = Q_Network(state_dim, action_dim).to(device)
    self.target_net_one = Q_Network(state_dim, action_dim).to(device)
    self.target_net_two = Q_Network(state_dim, action_dim).to(device)
    self.policy_net = Policy_Network(state_dim, action_dim).to(device)

    # Define the networks optimizers 
    self.optim_q_one = Adam( self.q_net_one.parameters(), lr=q_net_lr )
    self.optim_q_two = Adam( self.q_net_two.parameters(), lr=q_net_lr )
    self.optim_policy = Adam( self.policy_net.parameters(), lr=policy_lr )

    self.update_target_net()

  def play(self, state ) :  
    '''
    Core function for the action taking procces. Took the state and return random & determ action.

    @param state : Env state information.

    @return random_actions : These are action sampled randomly from distrib created by the policy.  
    @retrun deter_actions : There are action created with the mean of the distrib to be more deterministic.
    @return entropy : Entropy matix calcuate for the random_action selection.
    '''

    means, log_std = self.policy_net(state)
    standart_devs = torch.exp(log_std)
    action_distrib = Normal(means, standart_devs)
    random_samples = action_distrib.rsample()
    random_actions = torch.tanh(random_samples)
    deter_actions = torch.tanh(means) 

    log_probs_actions = action_distrib.log_prob(random_samples) - torch.log(1 - torch.pow(random_actions, 2) + self.epsilon)
    entropy = -torch.sum(log_probs_actions, dim=-1)

    return random_actions, deter_actions, entropy

  def act(self,state,  is_train) : 
    '''
    That return the either deterministic action or random action depending on what we want. 
    '''
    random_act, deter_act, _ = self.play(state)

    if is_train : 
      return random_act
    
    else : 
      return deter_act


  def learn(self, memory_sample):
    
    state, action, reward, next_state, is_over = zip(*memory_sample)
    
    state = torch.stack(state).detach()
    action = torch.stack(action).detach()
    reward = torch.stack(reward).detach()
    is_over = torch.stack(is_over).detach()
    next_state = torch.stack(next_state).detach()

    # Q-Networks Calculation    
    q_one_loss, q_two_loss = self.q_network_loss(state, action, reward, next_state, is_over)
    
    # Policy Network Calculation 
    policy_loss, entropy = self.policy_net_loss(state)

    # Update the each network depending on their loss
    self.networks_update(self.optim_q_one, self.q_net_one, q_one_loss)
    self.networks_update(self.optim_q_two, self.q_net_two, q_two_loss)
    self.networks_update(self.optim_policy, self.policy_net, policy_loss)

    return q_one_loss, q_two_loss, policy_loss, entropy

  def update_target_net(self, soft_update = False):
    
    if soft_update :
      for target, source in zip(self.target_net_one.parameters(), self.q_net_one.parameters()) : 
        target.data.copy_((1.0 - self.polyak) * target.data + self.polyak * source.data )

      for target, source in zip(self.target_net_one.parameters(), self.q_net_one.parameters()) : 
        target.data.copy_((1.0 - self.polyak) * target.data + self.polyak * source.data )
        
    else : 
      self.target_net_one.load_state_dict(self.q_net_one.state_dict())
      self.target_net_two.load_state_dict(self.q_net_two.state_dict())

  def q_network_target(self, reward, next_state, is_over) : 
    '''
    Calculate the Q-Networks target which will be useing on the loss function . 

    @param reward : Rewards samples come from replay buffer 
    @param next_state : Env state information sampled from replay buffer 
    @param is_over : Env execution information 
    '''
    with torch.no_grad() :
      action, _, entropy = self.play(next_state)

      target_value_one = torch.squeeze(self.target_net_one(next_state, action))
      target_value_two = torch.squeeze(self.target_net_two(next_state, action))
      
      next_q_val = torch.min(target_value_one, target_value_two)- self.alpha * entropy

    q_target = reward + self.gamma * (1 - is_over) * next_q_val  

    return q_target

  def q_network_loss(self, state, action, reward, next_state, is_over) : 
    '''
    Calculate the loss for each Q-Networks. 

    @param q_target : Target value for Q-Networks. 
    @param state : Env state info matrix
    @param action : Action sampled from Replay Memory 
    @return (q_one_loss, q_two_loss) : Loss value for each network
    '''    
    q_one_value = self.q_net_one(state, action)
    q_two_value = self.q_net_two(state, action)

    q_target = self.q_network_target(reward,next_state, is_over)
    
    q_one_loss = torch.mean( torch.square(q_one_value - q_target) )
    q_two_loss = torch.mean( torch.square(q_two_value - q_target) )
    
    return q_one_loss, q_two_loss
  
  def policy_net_loss(self, state): 

    action, _, entropy = self.play(state)
  
    with torch.no_grad() :
      q_value_one = self.q_net_one(state,action)
      q_value_two = self.q_net_two(state,action)


    policy_loss = torch.mean( torch.min( q_value_one, q_value_two ) - self.alpha * entropy )

    return policy_loss, torch.mean(entropy)

  def networks_update(self, optim, network, loss, grad_clip=None) : 
    '''
    Update the given networks depending on given loss. 
    
    @param optim : Should be optimizer of the given network. 
    @param network : Target Nn to update.
    @param loss : Calculated loss for network. 
    @param grad_clip : Gradient clip value if we are using. defult=None
    '''

    optim.zero_grad()
    loss.backward()

    # Apply the gradient clip if is set
    if grad_clip is not None : 
      for param in network.modules(): 
        nn.utils.clip_grad_norm_(param.parameters(), grad_clip)
    
    optim.step()      

  def save_models(self, model_path, is_avrg = '') : 
    
    torch.save(self.q_net_one.state_dict(), (model_path + is_avrg + 'car_q_net_one.pth'))
    torch.save(self.q_net_two.state_dict(), (model_path + is_avrg + 'car_q_net_two.pth'))
    torch.save(self.policy_net.state_dict(), (model_path + is_avrg + 'car_policy_net.pth'))

  def train_on(self,) : 
    '''
    We set the all the NN into train settings for the gradient
    '''

    self.q_net_one.train()
    self.q_net_two.train()
    self.policy_net.train()
    self.target_net_one.train()
    self.target_net_two.train()
  











