import random
import numpy as np


#napraviti metodu za resetirati zmiju
class Snake:

    DIRECTIONS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    #UP, RIGHT, DOWN, LEFT

    def __init__(self, start_position, start_length=3):
        self.current_direction_index = random.randrange(len(Snake.DIRECTIONS))
        self.alive = True

        self.my_blocks = [start_position]
        current_position = np.array(start_position)

        for _ in range(1, start_length):
            current_position = current_position - self.DIRECTIONS[self.current_direction_index]
            self.my_blocks.append(tuple(current_position))

    def step(self, action):
        if (action != self.current_direction_index) and (action != (self.current_direction_index + 2) % len(self.DIRECTIONS)):
            self.current_direction_index = action
        tail = self.my_blocks[-1]
        self.my_blocks = self.my_blocks[:-1]
        new_head = tuple(np.array(self.my_blocks[0]) + self.DIRECTIONS[self.current_direction_index])

        self.my_blocks = [new_head] + self.my_blocks
        return new_head, tail



class World:

    def __init__(self, size):  
        self.DEAD_REWARD = -1.0
        self.MOVE_REWARD = 0.0
        self.EAT_REWARD = 1.0
        
        self.EMPTY = 0
        self.FOOD = 1
        self.SNAKE_HEAD = 2
        self.SNAKE_BODY = 3
        self.WALL = 4

        self.DIRECTIONS = Snake.DIRECTIONS

        self.size = size
        self.world = np.zeros(size)
    
        self.world[0] = self.WALL
        self.world[size[0] - 1] = self.WALL
        self.world[:, 0] = self.WALL
        self.world[:, size[1] - 1] = self.WALL
        

        self.food_available_position = set(zip(*np.where(self.world == 0)))
        start_length = 3

        start_position = (random.randint(start_length, self.size[0] - start_length -1), \
                            random.randint(start_length, self.size[1] - start_length -1))

        
        self.snake = Snake(start_position, start_length)

        self.place_food()
        

    def place_food(self):
        available_positions = self.food_available_position.copy()
        available_positions -= set(self.snake.my_blocks)

        chosen_position = random.choice(list(available_positions))
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.apple_position = (chosen_position[0], chosen_position[1])

    def get_state(self):
        state = self.world.copy()
        for block in self.snake.my_blocks:
            state[block[0], block[1]] = self.SNAKE_BODY
        state[self.snake.my_blocks[0][0], self.snake.my_blocks[0][1]] = self.SNAKE_HEAD
        state /= 4
        return state

    def move_snake(self, action):

        new_food_needed = False
        new_head, old_tail = self.snake.step(action)
        if not(1 <= new_head[0] < self.size[0] - 1) or not(1 <= new_head[1] < self.size[1] - 1):
            self.snake.my_blocks = self.snake.my_blocks[1:]
            self.snake.alive = False
            reward = self.DEAD_REWARD
        elif new_head in self.snake.my_blocks[1:]:
            self.snake.alive = False
            reward = self.DEAD_REWARD
        if self.snake.alive and self.world[new_head[0], new_head[1]] == self.FOOD:
            self.world[new_head[0], new_head[1]] = 0
            self.snake.my_blocks.append(old_tail)
            new_food_needed = True
            reward = self.EAT_REWARD
        elif self.snake.alive:
            reward = self.MOVE_REWARD

        done = not self.snake.alive
        if new_food_needed:
            self.place_food()
        
        return reward, done

    
    def reset(self):
        self.world[self.apple_position[0], self.apple_position[1]] = 0
        start_length = 3
        start_position = (random.randint(start_length, self.size[0]-start_length - 1), \
                            random.randint(start_length, self.size[1] - start_length - 1))
        self.snake.my_blocks = [start_position]
        current_position = np.array(start_position)
        self.snake.current_direction_index = random.randrange(len(Snake.DIRECTIONS))
        for _ in range(1, start_length):
            current_position = current_position - self.snake.DIRECTIONS[self.snake.current_direction_index]
            self.snake.my_blocks.append(tuple(current_position))

        self.snake.alive = True
        self.place_food()

        return self.get_state()

