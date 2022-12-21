from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
import numpy as np
from collections import deque
import random



def build_dqn(lr, n_actions, input_shape):
    model = Sequential()

    model.add(Conv2D(
        16,
        kernel_size=(3,3),
        strides=(1, 1),
        padding="same",
        input_shape=(input_shape[0], input_shape[1], 1)
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3,3),
        strides=(1,1)
    ))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    #print(model.summary())

    return model



class Agent():
    def __init__(self, params):
        
        self.action_space = [i for i in range(params['n_actions'])]
        self.n_actions = params['n_actions']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.step_fix_target = params['step_fix_target']
        self.step_count = 0
        self.state_shape = params['state_shape']
        self.model_file = params['fname']
        
        self.memory = deque(maxlen=5000)

        if params['new_nn']:
            self.model = build_dqn(self.learning_rate, self.n_actions, self.state_shape)
            self.model.save(self.model_file)
        else:
            self.model = load_model(self.model_file)
        
        self.target_model = load_model(self.model_file)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            tmpState = np.copy(state)
            tmpState = tmpState.reshape(\
                        1, self.state_shape[0], self.state_shape[1], 1)
            actions = self.model.predict(tmpState)
            action = np.argmax(actions)

        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.step_count += 1
        if self.step_count == self.step_fix_target:
            self.step_count = 0
            self.model.save(self.model_file)
            self.target_model = load_model(self.model_file)

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        
        states = states.reshape(\
                    self.batch_size, self.state_shape[0], self.state_shape[1], 1)
        next_states = next_states.reshape(\
                    self.batch_size, self.state_shape[0], self.state_shape[1], 1)
        

        ind = np.array([i for i in range(self.batch_size)])

        
        targets = rewards + self.gamma * \
                    (np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


    