import numpy as np
import snake_game as sg
from render import Display
import time
from DQNAgent import Agent
import matplotlib.pyplot as plt


def main():
    params = dict()
    params['n_actions'] = 4
    params['gamma'] = 0.95
    params['epsilon'] = 1
    params['epsilon_decay'] = 0.995
    params['epsilon_min'] = 0.01
    params['learning_rate'] = 0.0005
    params['batch_size'] = 64
    params['step_fix_target'] = 500
    #(number of rows, number of columns)
    params ['state_shape'] = (10,10)
    #model's file name
    params['fname'] = "fixedQmodel"
    #True=make a new model, False=use an old model
    params['new_nn'] = True
    
    world = sg.World(params['state_shape'])
    display = Display(params['state_shape'])
    
    num_episodes = 5500
    max_steps = 1000

    agent = Agent(params)
    scores = []
    
    
    for i in range(num_episodes):
        state = world.reset()

        show_results = True

        done = False
        score = 0
        steps = 0
        while not done:
            display.refresh(world.apple_position, world.snake.my_blocks, \
                            world.snake.current_direction_index, score, i)
            action = agent.choose_action(state)
            reward, done = world.move_snake(action)
            score += reward
            if done:
                scores.append(score)
                print("{} ({})".format(score, i))
            next_state = world.get_state()

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            
            agent.learn()
            
            #time.sleep(0.2)
            steps += 1
            if steps > max_steps:
                scores.append(score)
                print("{} ({})".format(score, i))
                done = True
            
            if show_results and i % 500 == 0 and i != 0 and done:
                episode_numbers = np.arange(1, 501)
                plt.plot(episode_numbers, scores[-500:], 'o', color='black')
                plt.show()



if __name__ == '__main__':
    main()
    