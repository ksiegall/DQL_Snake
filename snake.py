# importing libraries
import time
import random
import numpy as np

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

        self.index_move = ["UP", "DOWN", "LEFT", "RIGHT"]

        self.available_moves = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        # use the reset function to initialize game state
        self.reset_env()

    def reset_env(self):
        # defining snake default position
        self.snake_position = [10, 5]

        # defining first 4 blocks of snake body
        self.snake_body = [[10, 5],
                            [9, 5],
                            [8, 5],
                            [7, 5]
                            ]
        # fruit position
        # make the initial fruit position close to the snake
        self.fruit_position = [random.randrange(self.snake_position[0], self.snake_position[0] + 5), random.randrange(self.snake_position[1] - 3, self.snake_position[1] + 3)]
        # self.fruit_position = [random.randrange(1, self.grid_size[0]), random.randrange(1, self.grid_size[1])]

        self.fruit_spawn = True

        # setting default snake direction towards
        # right
        self.direction = 'RIGHT'

        # initial score
        self.score = 0

    # displaying Score function
    def show_score(self, color, font, size):
    
        # creating font object score_font
        self.score_font = self.pygame.font.SysFont(font, size)
        
        # create the display surface object 
        # score_surface
        self.score_surface = self.score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text
        # surface object
        self.score_rect = self.score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(self.score_surface, self.score_rect)

    # game over function
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
            
            # deactivating self.pygame library
            # self.pygame.quit()
            
            print(f"Game over! Final score: {self.score}")
            # quit the program
            # quit()

        return False

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
        self.snake_body.insert(0, list(self.snake_position))
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 10
            self.fruit_spawn = False
        else:
            self.snake_body.pop()
            
        if not self.fruit_spawn:
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

        alive = True

        # Game Over conditions
        if self.snake_position[0] < 0 or self.snake_position[0] >= self.grid_size[0]:
            alive &= self.game_over()
            # you crashed into the wall
            self.score -= 2
        if self.snake_position[1] < 0 or self.snake_position[1] >= self.grid_size[1]:
            alive &= self.game_over()
            # you crashed into the wall
            self.score -= 2

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                alive &= self.game_over()
                
                # you crashed into yourself
                self.score -= 2

        if self.display_game:
            # displaying score continuously
            self.show_score(self.white, 'times new roman', 20)

            # Refresh game screen
            self.pygame.display.update()

            # Frame Per Second /Refresh Rate
            self.fps.tick(snake_speed)
        
        return alive

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
        grid = np.zeros(self.grid_size)
        grid[self.fruit_position[0], self.fruit_position[1]] = -1
        for pos in self.snake_body:
            # check for if the snake died by crossing the boundaries
            if pos[0] < 0 or pos[0] >= self.grid_size[0]:
                continue
            elif pos[1] < 0 or pos[1] >= self.grid_size[1]:
                continue
            else:
                grid[pos[0], pos[1]] = 1
        
        # print("body", self.snake_body[0][0], " ", self.snake_body[0][1])
        if self.snake_body[0][0] < 0 or self.snake_body[0][0] >= self.grid_size[0] or self.snake_body[0][1] < 0 or self.snake_body[0][1] >= self.grid_size[1]:
            pass    # again, fix up the head being elsewhere
        else:
            grid[self.snake_body[0][0], self.snake_body[0][1]] += 1
        grid = grid.reshape((1,self.grid_size[0]*self.grid_size[1]))
                
        # RETURN 
        return grid, self.score


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
                    
        game.step(command)
    
if __name__ == "__main__":
    main()