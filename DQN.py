# pytorch imlementation of DQN
import torch
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import threading
import time
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from snake import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

SNAKE_PIXEL_SIZE = 25
SNAKE_GRID = (25, 25)
snake = SnakeGame(SNAKE_GRID, cell_size_px=SNAKE_PIXEL_SIZE, display_game=False)
# ----- DQN parameters -----
n_actions = len(snake.index_move) # up, down, left, right
n_observations = snake.get_state()[0].size

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state_flat = state.view(state.size(0), -1)
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state_flat).max(1)[1].view(1, 1)
    else:
        # return a random action as a tensor
        action = random.randint(0, n_actions - 1)
        return torch.tensor([[action]], device=device, dtype=torch.long)  # size torch.Size([1, 1])

episode_durations = []
scores = []

fig, axes = plt.subplots(1, 2)

def plot_durations(show_result=False):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        axes[0].set_title('Result')
        axes[1].set_title('Scores - Result')
    else:
        axes[0].cla()
        axes[1].cla()
        # fig, axes = plt.subplots(1, 2)
        axes[0].set_title('Training...')
        axes[1].set_title('Scores - Training...')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Duration')
    axes[0].plot(durations_t.numpy(), color="blue")
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        axes[0].plot(means.numpy(), color="orange")

    scores_t = torch.tensor(scores, dtype=torch.float)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].plot(scores_t.numpy(), color="blue")

    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        score_means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        score_means = torch.cat((torch.zeros(99), score_means))
        axes[1].plot(score_means.numpy(), color="orange")

    fig.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
            
            
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # fix the dimensions of this batch (were 64, 1, 625), now (64, 625)
    state_batch_flat = state_batch.view(state_batch.size(0), -1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("state_batch_flat shape: ", state_batch_flat.shape)
    # print("action_batch shape", action_batch.shape)
    state_action_values = policy_net(state_batch_flat).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # fix that if it ended early there may be less end states
        if len(non_final_next_states) > 0:
            non_final_next_states_flat = non_final_next_states.view(non_final_next_states.size(0), -1)
            next_state_values[non_final_mask] = target_net(non_final_next_states_flat).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    

vis_thread_running = True
current_training_episode = 0 

def visualization_thread():
    # Create a separate instance of the snake game for visualization
    vis_snake = SnakeGame(SNAKE_GRID, cell_size_px=SNAKE_PIXEL_SIZE, display_game=True)
    vis_net = copy.deepcopy(policy_net)
    
    print("Visualization thread started")
    
    while vis_thread_running:
        # Reset the environment
        vis_snake.reset_env()
        alive = True
        
        # Get the initial state
        state, _ = vis_snake.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Update to latest model
        with torch.no_grad():
            vis_net.load_state_dict(policy_net.state_dict())
        model_from_episode = current_training_episode
        vis_snake.model_episode = model_from_episode
        
        # Play until the snake dies
        while alive:
            # Use the policy network to select an action (no exploration)
            with torch.no_grad():
                state_flat = state.view(state.size(0), -1)
                action = vis_net(state_flat).max(1)[1].view(1, 1)
            
            # Take the action
            alive, _ = vis_snake.step(action.item())
            
            # Get the new state
            if alive:
                observation, _ = vis_snake.get_state()
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
            # Check if model has been running too long (stuck in a loop probably)
            if current_training_episode - model_from_episode > 50:
                alive = False
                print("Model has been running too long (50 episodes), updating visualization...")


vis_thread = threading.Thread(target=visualization_thread)
vis_thread.daemon = True  
vis_thread.start()

checkpoint_interval = 250
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10000
else:
    num_episodes = 100

for i_episode in range(num_episodes):
    current_training_episode = i_episode  # Update the global episode counter
    
    # Initialize the environment and get its state
    snake.reset_env()
    state, _ = snake.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = select_action(state)
        alive, score = snake.step(action.item())
        observation, reward = snake.get_state()
        reward = torch.tensor([reward], device=device)  # TODO: add reward for dying LOL
        done = not alive

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state if next_state is not None else None

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            scores.append(score)
            plot_durations()
            if i_episode % checkpoint_interval == 0:
                torch.save(policy_net.state_dict(), f"dql_snake_episode_{i_episode}.pth")
            break

print('Complete')
vis_thread_running = False
plot_durations(show_result=True)
plt.ioff()
plt.show()