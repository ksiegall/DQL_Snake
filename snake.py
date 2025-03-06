# importing libraries
import time
import random
import numpy as np
import math

snake_speed = 15

# defining colors

# Initialising pygame

class SnakeGame():
    def __init__(self, grid_size: tuple[int, int], cell_size_px: int = 10, display_game: bool = True):
        self.grid_size = grid_size
        self.window_size_x = self.grid_size[0]*cell_size_px
        self.window_size_y = self.grid_size[1]*cell_size_px
        self.cell_size_px = cell_size_px

        # Choose if the game will be rendered or just played in the background
        self.display_game = display_game

        if self.display_game:
            import pygame
            self.pygame = pygame
            self.black = pygame.Color(0, 0, 0)
            self.white = pygame.Color(255, 255, 255)
            self.red = pygame.Color(255, 0, 0)
            self.green = pygame.Color(0, 255, 0)
            self.blue = pygame.Color(0, 0, 255)
            self.pygame.init()

            # Initialise game window
            self.pygame.display.set_caption('NerdsForNerds Sneak')
            self.game_window = self.pygame.display.set_mode((self.window_size_x, self.window_size_y))

            # FPS (frames per second) controller
            self.fps = self.pygame.time.Clock()

        self.index_move = ["UP", "LEFT", "DOWN", "RIGHT"]

        self.available_moves = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        self.FRUIT_SCORE_VAL = 10
        self.DEATH_SCORE_VAL = 0

        # use the reset function to initialize game state
        self.reset_env()
        
        # Add episode tracking attributes
        self.current_episode = 0
        self.model_episode = 0

    def reset_env(self):

        # setting default snake direction towards
        self.direction = random.choice(self.index_move)  # Choose a random direction from the list, excluding "LEFT"
        dir_of_motion = self.available_moves[self.direction]
        # defining snake default position
        self.snake_position = [10, 10]
        snake_len = 4 + int(random.random() > 0.5) + int(random.random() > 0.5)
        # defining first 4 blocks of snake body
        self.snake_body = [tuple(self.snake_position)]
        for i in range(snake_len-1):
            self.snake_body.append((self.snake_position[0]-(i+1)*dir_of_motion[0], self.snake_position[1]-(i+1)*dir_of_motion[1]))
        
        # fruit position
        # make the initial fruit position close to the snake
        self.fruit_position = tuple(self.snake_position)
        while self.fruit_position in self.snake_body:
            self.fruit_position = (random.randrange(self.snake_position[0] - 3, self.snake_position[0] + 3), 
                                   random.randrange(self.snake_position[1] - 3, self.snake_position[1] + 3))
        # self.fruit_position = [random.randrange(1, self.grid_size[0]), random.randrange(1, self.grid_size[1])]
        self.fruit_spawn = True

        # initial score
        self.score = 0

        self.time = 1
        self.time_of_last_fruit = self.time

        self.alive = True

    # displaying Score function
    def show_score(self, color, font, size):
    
        # creating font object score_font
        self.score_font = self.pygame.font.SysFont(font, size)
        
        # create the display surface object for score
        self.score_surface = self.score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text surface object
        self.score_rect = self.score_surface.get_rect()
        
        # displaying score text
        self.game_window.blit(self.score_surface, self.score_rect)
        
        # create the display surface object for model episode info
        episode_text = f'Model from Episode: {self.model_episode}'
        self.episode_surface = self.score_font.render(episode_text, True, color)
        
        # create a rectangular object for the episode text
        self.episode_rect = self.episode_surface.get_rect()
        self.episode_rect.topright = (self.window_size_x - 10, 10)  # Position in top right corner
        
        # displaying episode text
        self.game_window.blit(self.episode_surface, self.episode_rect)

    def game_over(self):
        
        if self.display_game:
            # creating font object my_font
            my_font = self.pygame.font.SysFont('times new roman', 50)
            
            # creating a text surface on which text 
            # will be drawn
            game_over_surface = my_font.render(
                'Your Score is : ' + str(self.score), True, self.red)
            
            # create a rectangular object for the text 
            # surface object
            
            # setting position of the text
            game_over_rect = game_over_surface.get_rect()
            game_over_rect.midtop = (self.window_size_x/2, self.window_size_y/4)
            # blit will draw the text on screen
            self.game_window.blit(game_over_surface, game_over_rect)
            self.pygame.display.flip()
            
            # after 0.5 seconds we will quit the program
            time.sleep(0.5)
            
            # print(f"Game over! Final score: {self.score}") 

    def step(self, command: str | int):
        if isinstance(command, int):
            # convert int command to str
            # print(f"Converting {command} to a string command")
            command = self.index_move[command]
        self.direction = self.get_new_direction(command)
        
        # Moving the snake
        motion = self.available_moves[self.direction]
        self.snake_position[0] += motion[0]
        self.snake_position[1] += motion[1]
        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        self.snake_body.insert(0, tuple(self.snake_position))
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += self.FRUIT_SCORE_VAL
            self.time_of_last_fruit = self.time
            self.fruit_spawn = False
        else:
            self.snake_body.pop()
            
        if not self.fruit_spawn:

            # while self.fruit_position in self.snake_body:
            self.fruit_position = [random.randrange(0, self.grid_size[0]), 
                            random.randrange(0, self.grid_size[1])]
            self.fruit_spawn = True

        if self.display_game:
            self.game_window.fill(self.black)
            
            for pos in self.snake_body:
                self.pygame.draw.rect(self.game_window, self.green,
                                self.pygame.Rect(pos[0]*self.cell_size_px, pos[1]*self.cell_size_px, self.cell_size_px, self.cell_size_px))
            self.pygame.draw.rect(self.game_window, self.white, self.pygame.Rect(
                self.fruit_position[0]*self.cell_size_px, self.fruit_position[1]*self.cell_size_px, self.cell_size_px, self.cell_size_px))

        self.alive = True

        # Game Over conditions
        if self.check_death_condition(self.snake_position):
            self.alive = False
            self.game_over()
            self.score -= self.DEATH_SCORE_VAL
            # you died

        if self.display_game:
            # displaying score continuously
            self.show_score(self.white, 'times new roman', 20)

            # Refresh game screen
            self.pygame.display.update()

            # Frame Per Second /Refresh Rate
            self.fps.tick(snake_speed)
        
        self.time += 1

        return self.alive, self.score
    
    def check_death_condition(self, snake_pos: tuple[int, int]):
        if snake_pos[0] < 0 or snake_pos[0] >= self.grid_size[0]:
            # you crashed into the wall
            return True
        if snake_pos[1] < 0 or snake_pos[1] >= self.grid_size[1]:
            # you crashed into the wall
            return True

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:   
                # you crashed into yourself
                return True
        
        # No crash condition
        return False

    def get_new_direction(self, attempted_direction):
        if attempted_direction == 'UP' and self.direction != 'DOWN':
            return 'UP'
        if attempted_direction == 'DOWN' and self.direction != 'UP':
            return 'DOWN'
        if attempted_direction == 'LEFT' and self.direction != 'RIGHT':
            return 'LEFT'
        if attempted_direction == 'RIGHT' and self.direction != 'LEFT':
            return 'RIGHT'
        return self.direction

    def get_state(self):
        # RETURN GRID
        # EMPTY = 0
        # SNAKE = 1
        # SNAKE HEAD = 2
        # APPLE = -1

        # --- FEATURES ---

        #                      STRAIGHT                TURN LEFT (CCW)                                         TURN RIGHT (CW)
        death_test_dirs = [self.direction, self.index_move[(self.index_move.index(self.direction)+1) % len(self.index_move)], self.index_move[self.index_move.index(self.direction)-1]]
        death_test_results = []

        for direction in death_test_dirs:
            motion = self.available_moves[direction]
            new_x = self.snake_position[0] + motion[0]
            new_y = self.snake_position[1] + motion[1]
            death_test_results.append(self.check_death_condition((new_x, new_y)))

        # One-hot representation of snake current direction
        cur_direction = [direction == self.direction for direction in self.index_move]
        
#TODO: !!! THIS DIRECTION also SCALES with the distance???
        direction_to_apple = self.fruit_position[0] - self.snake_position[0], self.fruit_position[1] - self.snake_position[1]

        dist_to_apple = abs(direction_to_apple[0]) + abs(direction_to_apple[1])+1
        
        # apple_dir = [max(-direction_to_apple[1]/dist_to_apple,0), # UP
        #              max(-direction_to_apple[0]/dist_to_apple,0), # LEFT
        #              max(direction_to_apple[1]/dist_to_apple,0),  # DOWN
        #              max(direction_to_apple[0]/dist_to_apple,0)]  # RIGHT
        
        # one hot representation of the direction to the apple in X and Y w.r.t. to the snake current direction
        apple_dir = [0, 0, 0, 0]
        
        # if the apple is to the UP of the current snake position    
        apple_dir[0] = 1 if direction_to_apple[1] > 0 else 0
        # if the apple is to the LEFT of the current snake position
        apple_dir[1] = 1 if direction_to_apple[0] < 0 else 0
        # if the apple is to the DOWN of the current snake position
        apple_dir[2] = 1 if direction_to_apple[1] < 0 else 0
        # if the apple is to the RIGHT of the current snake position
        apple_dir[3] = 1 if direction_to_apple[0] > 0 else 0
        
        body_length_norm = len(self.snake_body) / (self.grid_size[0] * self.grid_size[1])

        features = np.hstack((death_test_results, cur_direction, apple_dir, body_length_norm))
        
        # # --- make the entire grid a feature ---
        # grid = np.zeros(self.grid_size)
        # grid[self.fruit_position[0], self.fruit_position[1]] = -1
        # for pos in self.snake_body:
        #     # check for if the snake died by crossing the boundaries
        #     if pos[0] < 0 or pos[0] >= self.grid_size[0]:
        #         continue
        #     elif pos[1] < 0 or pos[1] >= self.grid_size[1]:
        #         continue
        #     else:
        #         grid[pos[0], pos[1]] = 1
        
        # # print("body", self.snake_body[0][0], " ", self.snake_body[0][1])
        # if self.snake_body[0][0] < 0 or self.snake_body[0][0] >= self.grid_size[0] or self.snake_body[0][1] < 0 or self.snake_body[0][1] >= self.grid_size[1]:
        #     pass    # again, fix up the head being elsewhere
        # else:
        #     grid[self.snake_body[0][0], self.snake_body[0][1]] += 1
        # grid = grid.reshape((1,self.grid_size[0]*self.grid_size[1]))
                
        # --- REWARDS ---
        time_weight = 0.1
        # RETURN features, score + add punishment for not collecting apple
        reward = math.log(self.time) - time_weight*(self.time - self.time_of_last_fruit)

        if self.time - self.time_of_last_fruit == 1:
            reward += 10

        if not self.alive:
            reward -= 50
        
        # reward it a tiny bit for getting closer to the apple or not
        # it will be either +1, 0, or -1 in direction divided by distance (so the closer we are the more does this reward actually matter)
        direction_to_apple = self.fruit_position[0] - self.snake_position[0], self.fruit_position[1] - self.snake_position[1]
        # make it be the difference invariant of the direction the snake is facing
        #   with +X being forward, and +Y being right relative to snake motion
        if self.direction == "UP":
            reward -= min(direction_to_apple[1]/(dist_to_apple + 1), 0)
        elif self.direction == "LEFT":
            reward -= min(direction_to_apple[0]/(dist_to_apple + 1), 0)
        elif self.direction == "DOWN":
            reward += max(direction_to_apple[1]/(dist_to_apple + 1), 0)
        elif self.direction == "RIGHT":
            reward += max(direction_to_apple[0]/(dist_to_apple + 1), 0)
        
        # # use the sign of them to see if we are heading towards an apple, or away from it in X and Y
        # reward += np.sign(direction_to_apple[0]) / (dist_to_apple + 1)
        # reward += np.sign(direction_to_apple[1]) / (dist_to_apple + 1)
        
        return features, reward


def main():
    # KEYBOARD CONTROL OF SNAKE GAME
    import pygame
    game = SnakeGame((72,48), cell_size_px=25, display_game=True)
    command = "RIGHT"
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
                    
        alive, _ = game.step(command)

        if not alive:
            break
    
if __name__ == "__main__":
    main()