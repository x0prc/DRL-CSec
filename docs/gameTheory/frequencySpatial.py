import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

class CRNEnv(gym.Env):
    def __init__(self, num_channels, num_jammers):
        self.num_channels = num_channels
        self.num_jammers = num_jammers
        
        self.action_space = gym.spaces.Discrete(num_channels)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_channels, num_channels, 2))
        
        self.current_channel = 0
        self.current_sinr = 0
        
        self.reward = 0
        self.done = False
        
    def step(self, action):
        self.current_sinr = self.calculate_sinr(action)
        
        # Calculate the reward based on the SINR
        self.reward = self.current_sinr
        
        # Update the spectrum waterfall
        self.update_spectrum_waterfall()
        
        state = self.spectrum_waterfall
        
        return state, self.reward, self.done, {}
    
    def reset(self):
        self.current_channel = 0
        self.current_sinr = 0
        self.reward = 0
        self.done = False
        
        self.spectrum_waterfall = np.zeros((self.num_channels, self.num_channels, 2))
        
        return self.spectrum_waterfall
    
    def calculate_sinr(self, action):
        self.current_channel = action
        signal_power = self.signal_power[action]
        jammer_power = np.sum(self.jammer_power[action, :])
        sinr = signal_power / (jammer_power + self.noise_power)
        return sinr
        pass
    
    def update_spectrum_waterfall(self):
        self.spectrum_waterfall[:, :-1, 0] = self.spectrum_waterfall[:, 1:, 0]
        self.spectrum_waterfall[:, :-1, 1] = self.spectrum_waterfall[:, 1:, 1]
        
        self.spectrum_waterfall[:, -1, 0] = self.signal_power / np.max(self.signal_power)
        self.spectrum_waterfall[:, -1, 1] = np.sum(self.jammer_power, axis=1) / np.max(np.sum(self.jammer_power, axis=1))
        pass

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = CRNEnv(num_channels=10, num_jammers=5)
agent = DQNAgent(state_size=env.observation_space.shape, action_size=env.action_space.n)

for episode in range(1000):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode {episode} finished after {time+1} timesteps")
            break
        if len(agent.memory) > 32:
            agent.replay(32)