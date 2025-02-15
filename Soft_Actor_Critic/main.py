
# Import Section
import yaml
import torch 
import argparse
import gymnasium as gym 
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from utils import Replay_Memory, Avrg_Reward_Que

with open('conf.yaml', 'r') as file : 
  config = yaml.safe_load(file)


alpha = config['Agent']['alpha']
gamma = config['Agent']['gamma']
polyak = config['Agent']['polyak']
epsilon = config['Agent']['epsilon']
q_net_lr = config['Agent']['q_net_lr']
policy_lr = config['Agent']['policy_lr']

batch_size = config['Train']['batch_size']
memory_size = config['Train']['memory_size']
epoch_number = config['Train']['epoch_number']
exploration_limit = config['Train']['exploration_limit']
target_update_freq = config['Train']['target_update_freq']

log_path = config['Logging']['log_path']
model_path = config['Logging']['model_path']



def Train() : 

  # Create the tensorboard writer
  tensorboard_writer = SummaryWriter()

  # Check if the GPU is available or not and set the device accordingly
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  env  = gym.make("BipedalWalker-v3", hardcore=True, max_episode_steps=2500)
  state_space  = env.observation_space.shape[0]
  aciton_space = env.action_space.shape[0]

  agent = Agent(state_space, aciton_space, alpha, epsilon, gamma, q_net_lr, policy_lr, device, polyak)
  agent.train_on()

  replay_memory = Replay_Memory(memory_size, batch_size)

  avg_reward = Avrg_Reward_Que(20)
  best_reward = float('-inf')

  step = 0

  for episode in range(epoch_number)  :

    state, _ = env.reset() 
    episode_reward = 0
    is_episode_over = False
    state = torch.tensor(state, dtype=torch.float32, device=device).clone()

    while not is_episode_over : 
      
      if step < exploration_limit : 
        action = env.action_space.sample()
        action = torch.tensor(action, dtype=torch.float32, device=device)
        step += 1 

      else : 
        action = agent.act(state, True)

      next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())

      episode_reward += reward
      is_episode_over = terminated or truncated

      state = torch.tensor(state, dtype=torch.float32, device=device, )
      reward = torch.tensor(reward, dtype=torch.float32, device=device, )
      next_state = torch.tensor(next_state, dtype=torch.float32, device=device, )
      is_over = torch.tensor(is_episode_over, dtype=torch.float32, device=device, )

      replay_memory.push(state, action, reward, next_state, is_over)

      state = next_state
      env.render()
    ## Game loop is over in here 

    avg_reward.push(episode_reward)
    avrg_reward = avg_reward.avrg_reward()

    tensorboard_writer.add_scalar('Episode/Reward', episode_reward, episode)
    tensorboard_writer.add_scalar('Episode/Avg Reward', avrg_reward, episode)
    tensorboard_writer.add_scalar('Episode/Steps', step, episode)

    
    if avg_reward.is_model_better() : 
      agent.save_models(model_path, 'Avrg_')
      print(f"Model saved because avrg reward. Episode : {episode}, Reward : {avrg_reward}")

    if episode_reward > best_reward : 
      agent.save_models(model_path)
      best_reward = episode_reward
      print(f"Model saved because best reward. Episode : {episode}, Reward : {episode_reward}")


    if len(replay_memory.memory) > batch_size : 
      memory_sample = replay_memory.sample()
      q_loss_one, q_loss_two, policy_loss, entropy = agent.learn(memory_sample)

      tensorboard_writer.add_scalar('Loss/Q Net One', q_loss_one, episode)
      tensorboard_writer.add_scalar('Loss/Q Net Two', q_loss_two, episode)
      tensorboard_writer.add_scalar('Loss/Policy', policy_loss, episode)
      tensorboard_writer.add_scalar('Loss/Entropy', torch.mean(entropy), episode)


    
    agent.update_target_net(soft_update=True)
      


def Play() : 
  pass



def Main(input_args) : 

  if input_args.mode == 'Train' : 
    Train()

  elif input_args.mode == 'Test' : 
    Play()

if __name__ == '__main__' :

  input_arg = argparse.ArgumentParser(description= "Input argument passer")
  input_arg.add_argument('--mode', type=str, help="Set the Train or Test mode", default= 'Train')

  print("Main function is called")
  Main(input_arg.parse_args())















