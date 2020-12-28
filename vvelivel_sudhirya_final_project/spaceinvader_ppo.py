import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

from torch.nn import functional as F
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        memsize = 256
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)

        # The second convolution layer takes a
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        # The third convolution layer takes a
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.actions = nn.Linear(in_features=512, out_features=4)
        self.soft = nn.Softmax(dim=-1)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)
        
    def forward(self, obs):
        #print(obs.shape)
        #obs = obs.unsqueeze(0)
        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = F.relu(self.lin(h))

        actions = self.soft(self.actions(h))
        value = self.value(h).reshape(-1)
        return actions, value
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs, _ = self.forward(state.unsqueeze(0))
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        #print(f'evaluate {state.shape}')
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        #state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.max_grad_norm = .5
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        total_loss = 0

        print(len(old_states))
        steps = len(old_states)
        b_inds = np.arange(steps)
        self.minibatch_steps = 8

        # Optimize policy for K epochs:
        for start in range(0, steps, self.minibatch_steps): #range(self.K_epochs):
            mb_inds = b_inds[start:start + self.minibatch_steps]
            mb_inds = torch.from_numpy(mb_inds).long().cuda()
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[mb_inds], old_actions[mb_inds])
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs[mb_inds].detach())
                
            # Finding Surrogate Loss:
            advantages = rewards[mb_inds] - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards[mb_inds]) - 0.01*dist_entropy
            total_loss += loss.mean().item()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            #torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        print(f' loss {total_loss/self.K_epochs}')
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
import cv2
import numpy as np
class Game:
    """
    ## <a name="game-environment"></a>Game environment
    This is a wrapper for OpenAI gym game environment.
    We do a few things here:
    1. Apply the same action on four frames and get the last frame
    2. Convert observation frames to gray and scale it to (84, 84)
    3. Stack four frames of the last four actions
    4. Add episode information (total reward for the entire episode) for monitoring
    5. Restrict an episode to a single life (game has 5 lives, we reset after every single life)
    #### Observation format
    Observation is tensor of size (4, 84, 84). It is four frames
    (images of the game screen) stacked on first axis.
    i.e, each channel is a frame.
    """

    def __init__(self, seed: int):
        # create environment
        self.env = gym.make('SpaceInvaders-v4')
        self.env.seed(seed)

        # tensor for a stack of 4 frames
        self.obs_4 = np.zeros((4, 84, 84))

        # keep track of the episode rewards
        self.rewards = []
        # and number of lives left
        self.lives = 0

    def step(self, action):
        """
        ### Step
        Executes `action` for 4 time steps and
         returns a tuple of (observation, reward, done, episode_info).
        * `observation`: stacked 4 frames (this frame and frames for last 3 actions)
        * `reward`: total reward while the action was executed
        * `done`: whether the episode finished (a life lost)
        * `episode_info`: episode information if completed
        """

        reward = 0.
        done = None

        # run for 4 steps
        for i in range(4):
            # execute the action in the OpenAI Gym environment
            obs, r, done, info = self.env.step(action)

            reward += r

            # get number of lives left
            lives = self.env.unwrapped.ale.lives()
            # reset if a life is lost
            if lives < self.lives:
                done = True
                break

        # Transform the last observation to (84, 84)
        obs = self._process_obs(obs)

        # maintain rewards for each step
        self.rewards.append(reward)

        if done:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
            # get the max of last two frames
            # obs = self.obs_2_max.max(axis=0)

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info
    def render(self):
        return self.env.render(mode='rgb_array')  

    def reset(self):
        """
        ### Reset environment
        Clean up episode info and 4 frame stack
        """

        # reset OpenAI Gym environment
        obs = self.env.reset()

        # reset caches
        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4

    @staticmethod
    def _process_obs(obs):
        """
        #### Process game frames
        Convert game frames to gray and rescale to 84x84
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs/255.
import scipy.misc

def myplt(inputs,lbl="Reward") :
  print(len(inputs))
  plt.plot(range(len(inputs)), inputs)
  plt.ylabel(lbl)
  plt.xlabel("Episodes")
  plt.savefig(f"{lbl}.png",dpi=300)#show()

def main():
    ############## Hyperparameters ##############
    env_name = "SpaceInvaders-v4"
    # creating environment
    env = gym.make(env_name)
    state_dim = 1 #env.observation_space.shape[0]
    print(state_dim)
    action_dim = 4
    render = False
    solved_reward = 1000         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 500000        # max training episodes
    max_timesteps = 10000         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 32      # update policy every n timesteps
    lr = 2.5e-4
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 32                # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    game = Game(112)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    all_running_rewards = []
    highest_running_reward = 0
    
    state = game.reset()
    # training loop
    for i_episode in range(1, max_episodes+1):
        
        #print(f'state {state.shape}')
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = game.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            '''if timestep % update_timestep == 0:
                
                timestep = 0'''
            
            running_reward += reward
            #if render:
            #img = game.render()
            #scipy.misc.imsave(f'outfile_{t}.jpg', img)
            #print(type(img))
            if done:
                print(f"episode done. {timestep}")
                ppo.update(memory)
                memory.clear_memory()
                state = game.reset()
                timestep = 0
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (highest_running_reward):
            #print("########## Solved! ##########")
            highest_running_reward = running_reward
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            #break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            all_running_rewards.append(running_reward)
            myplt(all_running_rewards, lbl=f'{env_name}_Reward')

            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
