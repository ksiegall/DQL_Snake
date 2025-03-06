from DQN import *


def demo_qlearn(checkpoint_path: str):
        # Create a separate instance of the snake game for visualization
        game = SnakeGame(SNAKE_GRID, cell_size_px=SNAKE_PIXEL_SIZE, display_game=True)
        policy_net.load_state_dict(torch.load(checkpoint_path))
        
        print("Demo started")
        
        while True:
            # Reset the environment
            game.reset_env()
            alive = True
            
            # Get the initial state
            state, _ = game.get_state()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Play until the snake dies
            while alive:
                # Use the policy network to select an action (no exploration)
                with torch.no_grad():
                    state_flat = state.view(state.size(0), -1)
                    action = policy_net(state_flat).max(1)[1].view(1, 1)
                
                # Take the action
                alive, _ = game.step(action.item())
                
                # Get the new state
                if alive:
                    observation, _ = game.get_state()
                    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    
if __name__ == "__main__":
    trained_model_path = "checkpoints/dql_snake_episode_9750.pth"
    demo_qlearn(trained_model_path)