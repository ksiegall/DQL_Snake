from snake import *
from DQN import *
import pickle
import pygame
from datetime import datetime

def create_supervised_example():
    # KEYBOARD CONTROL OF SNAKE GAME
    memory = ReplayMemory(10000)
    SNAKE_PIXEL_SIZE = 25
    SNAKE_GRID = (25, 25)
    game = SnakeGame(SNAKE_GRID, cell_size_px=SNAKE_PIXEL_SIZE, display_game=True)
    command = "RIGHT"
    state, _ = game.get_state()
    # Main Function
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
                    
        action = game.index_move.index(command)
        alive, score = game.step(action)
        next_state, reward = game.get_state()

        memory.push(state, action, next_state, reward)
        
        state = next_state

        pygame.time.delay(100)

        if not alive:
            now = datetime.now()
            timestamp_string = now.strftime("%m_%d_%H_%M_%S")
            filename = f"supervised_examples/Score_{score}_{timestamp_string}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(memory, file)
            break

if __name__ == "__main__":
    create_supervised_example()