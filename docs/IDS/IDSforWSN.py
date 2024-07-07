import gym
import numpy as np
from collections import deque
from skfuzzy import control as ctrl

class WSNEnv(gym.Env):
    def __init__(self, num_agents, num_actions):
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        self.action_space = [gym.spaces.Discrete(num_actions) for _ in range(num_agents)]
        self.observation_space = [gym.spaces.Box(low=0, high=1, shape=(num_agents + 1,)) for _ in range(num_agents)]
        
        self.attack_detected = False
        self.detection_accuracy = 0
        self.energy_consumption = 0
        self.network_lifetime = 0
        
    def step(self, actions):
        states = []
        rewards = []
        for agent_id, action in enumerate(actions):
            self.update_environment(agent_id, action)
            
            # Calculate the reward for the agent
            reward = self.calculate_reward(agent_id)
            rewards.append(reward)
            
            state = self.update_state(agent_id)
            states.append(state)
        
        return states, rewards, self.attack_detected, {}
    
    def reset(self):
        self.attack_detected = False
        self.detection_accuracy = 0
        self.energy_consumption = 0
        self.network_lifetime = 0
        
        return [np.zeros(self.num_agents + 1) for _ in range(self.num_agents)]
    
    def calculate_reward(self, agent_id):
        # Calculate the reward for the agent based on the detection accuracy, energy consumption, and network lifetime
        return self.detection_accuracy - self.energy_consumption + self.network_lifetime
    
    def update_state(self, agent_id):
        # Update the state for the agent based on the current environment and the actions of other agents
        return np.concatenate((np.array([self.attack_detected]), np.random.rand(self.num_agents)))

class FuzzyQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        
        self.attack_detected = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'attack_detected')
        self.action = ctrl.Consequent(np.arange(0, self.action_size, 1), 'action')
        
        # Define the membership functions for the fuzzy system
        self.attack_detected['low'] = ctrl.membership.gaussmf(self.attack_detected.universe, 0.2, 0.1)
        self.attack_detected['medium'] = ctrl.membership.gaussmf(self.attack_detected.universe, 0.5, 0.1)
        self.attack_detected['high'] = ctrl.membership.gaussmf(self.attack_detected.universe, 0.8, 0.1)
        
        self.action['low'] = ctrl.membership.gaussmf(self.action.universe, 1, 1)
        self.action['medium'] = ctrl.membership.gaussmf(self.action.universe, 3, 1)
        self.action['high'] = ctrl.membership.gaussmf(self.action.universe, 5, 1)
        
        # Define the fuzzy rules
        rule1 = ctrl.Rule(self.attack_detected['low'], self.action['low'])
        rule2 = ctrl.Rule(self.attack_detected['medium'], self.action['medium'])
        rule3 = ctrl.Rule(self.attack_detected['high'], self.action['high'])
        
        self.fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.fuzzy_agent = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        self.fuzzy_agent.input['attack_detected'] = state[0]
        self.fuzzy_agent.compute()
        return int(self.fuzzy_agent.output['action'])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.max(self.act(next_state)))
            self.fuzzy_agent.input['attack_detected'] = state[0]
            self.fuzzy_agent.compute()
            self.fuzzy_agent.output['action'] = target
            self.fuzzy_agent.commit()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = WSNEnv(num_agents=3, num_actions=5)
agents = [FuzzyQAgent(env.observation_space[i].shape[0], env.action_space[i].n) for i in range(env.num_agents)]

for episode in range(1000):
    states = env.reset()
    done = False
    while not done:
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        next_states, rewards, done, _ = env.step(actions)
        for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, next_states):
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)
        states = next_states