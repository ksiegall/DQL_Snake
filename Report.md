# Title
**Contributers:** K. Seigall (krsiegall@wpi.edu), Jakub Jandus (jjandus@wpi.edu), Ava Chadbourne (agchadbourne@wpi.edu)

## Introduction

The goal of our final project was to train a model using Deep Q Learning (DQL) to play Snake. Accuracy of the model was measuered using the final game score; higher scores means a better model. We chose this issue because one of our team members loves to play Snake, and we thought it would be interesting to tackle a simple game with DQL.

The model was given: actions, the game board, (**ADD LATER**). The actions that could be taken were: UP, DOWN, LEFT, RIGHT

## Methods

**Initial Model**: We just converted the model given in the tutorial from [Paszke and Towers](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). The reward was just the score of the game.

**Visualize the Game**: Now run a visual representation of the model's current skill while it's still training, so you can always see a game running.

**Cost of Living**: Added cost of living penalty to reward (so snake won't just go in circles forever). It commits suicide sometimes.

**Changed State Information**: Instead of sending the entire grid, we now send information about what directions would immediately lead to death, the direction of the apple, the current direction of the snake, and the snake's body length (normalized). Reward is now based on distance from apple.

**Reward Change, Changed Snake Spawn Code**: Snake can now spawn in any orientation. Reward tweaked so snake is rewarded for staying alive, but the longer the game goes on the more it is penalized for taking awhile to find the apple.

**Reward Change: higher reward for moving toward apple**: Reward snake for moving closer and closer to apple. No punishment for moving away (yet)



## Table of Results
**Initial Model**: Kinda shit

**Cost of Living**: Commits suicide sometimes.

**Changed State Info**: Hey it kinda works. Snake sometimes wiggles in the corner in circles.

**Reward, Snake Spawn Change**: Works, but no huge improvements

**Higher reward for moving toward apple**: Much better

## Conclusions

## References
[Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) by Adam Paszke, Mark Towers.
