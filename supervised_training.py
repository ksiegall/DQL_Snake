from snake import *
from DQN import *
import pickle
import pygame
from datetime import datetime
import os

def create_supervised_example():
    # KEYBOARD CONTROL OF SNAKE GAME
    memory = ReplayMemory(10000)
    SNAKE_PIXEL_SIZE = 25
    SNAKE_GRID = (25, 25)
    game = SnakeGame(SNAKE_GRID, cell_size_px=SNAKE_PIXEL_SIZE, display_game=True)
    command = "RIGHT"
    state, _ = game.get_state()
    # Main Function
    delay = 100
    while True:
        # handling key events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    command = 'UP'
                if event.key == pygame.K_DOWN:
                    command = 'DOWN'
                if event.key == pygame.K_LEFT:
                    command = 'LEFT'
                if event.key == pygame.K_RIGHT:
                    command = 'RIGHT'
                if event.key == pygame.K_SPACE:
                    # Set game to slow motion so play can make best move
                    delay = 500
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    # Set game to slow motion so play can make best move
                    delay = 250
        action = game.index_move.index(command)
        alive, score = game.step(action)
        observation, reward = game.get_state()
        reward = torch.tensor([reward], device=device)  # TODO: add reward for dying LOL

        if not alive:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        
        state = next_state

        pygame.time.delay(delay)

        if not alive:
            now = datetime.now()
            timestamp_string = now.strftime("%m_%d_%H_%M_%S")
            filename = f"supervised_examples/Score_{score}_{timestamp_string}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(memory, file)
            break


                
def optimize_model(memory):
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
    non_final_next_states = torch.cat([torch.from_numpy(s).to(device=device) if type(s) != torch.Tensor else s for s in batch.next_state if s is not None])
    # print(batch.state)
    state_batch = torch.cat([torch.from_numpy(s).to(device=device).unsqueeze(0) if type(s) != torch.Tensor else s for s in batch.state if s is not None])
    action_batch = torch.cat([torch.tensor([a], device=device).unsqueeze(0) for a in batch.action])
    reward_batch = torch.cat([torch.tensor(r, device=device).unsqueeze(0) for r in batch.reward])
    
    # fix the dimensions of this batch (were 64, 1, 625), now (64, 625)
    state_batch_flat = state_batch.view(state_batch.size(0), -1).to(dtype=torch.float32)

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

def train_from_examples(example_folder):

    examples = os.listdir(example_folder)
    for index, example in enumerate(examples):
        print(f"Training Supervised Example {index+1}: {example}")
        with open(example_folder+example, "rb") as supervised_example:
            memory = pickle.load(supervised_example)
            print(f"Round Length: {len(memory)}")
            for i in range(len(memory)):
                optimize_model(memory)
            
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
    
    torch.save(policy_net.state_dict(), f"checkpoints/dql_snake_post_supervision.pth")

if __name__ == "__main__":
    # create_supervised_example()
    train_from_examples("./supervised_examples/")