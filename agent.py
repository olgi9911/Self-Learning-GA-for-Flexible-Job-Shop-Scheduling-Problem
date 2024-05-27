import numpy as np
import pandas as pd
import random

class Agent:
    def __init__(self, num_states=20, num_actions=10, epsilon=0.85, learning_rate=0.75, gamma=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.q_table_pc = np.zeros((self.num_states, self.num_actions))
        self.q_table_pm = np.zeros((self.num_states, self.num_actions))

    def select_action_pc(self, state):
        if random.random() < self.epsilon:
           action_idx = np.argmax(self.q_table_pc[state])
        else:
            action_idx = random.randint(0, self.num_actions - 1)

        return action_idx
    
    def select_action_pm(self, state):
        if random.random() < self.epsilon:
           action_idx = np.argmax(self.q_table_pm[state])
        else:
            action_idx = random.randint(0, self.num_actions - 1) 

        return action_idx
    
    def learn(self, q_table, state, action, reward, next_state, next_action, is_SARSA=True):
        current_q = q_table[state, action]
        if is_SARSA:
            next_q = q_table[next_state, next_action]
        else:
            next_q = np.max(q_table[next_state])
            
        q_table[state, action] = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q)
        #q_table[state, action] = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.gamma * next_q)

    def learn_pc(self, state, action, reward, next_state, next_action, is_SARSA=True):
        self.learn(self.q_table_pc, state, action, reward, next_state, next_action, is_SARSA)

    def learn_pm(self, state, action, reward, next_state, next_action, is_SARSA=True):
        self.learn(self.q_table_pm, state, action, reward, next_state, next_action, is_SARSA)