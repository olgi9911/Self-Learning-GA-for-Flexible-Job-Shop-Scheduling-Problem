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
           action_idx = np.argmax(self.q_table_pc[state, :])
        else:
            action_idx = random.randint(0, self.num_actions - 1)

        return action_idx
    
    def select_action_pm(self, state):
        if random.random() < self.epsilon:
           action_idx = np.argmax(self.q_table_pm[state, :])
        else:
            action_idx = random.randint(0, self.num_actions - 1) 

        return action_idx
    
    def learn_pc(self, state, action, reward, next_state, next_action, is_SARSA=True):
        if is_SARSA:
            self.q_table_pc[state, action] = (1 - self.learning_rate) * self.q_table_pc[state, action] + self.learning_rate * (reward + self.gamma * self.q_table_pc[next_state, next_action])
        else:
            self.q_table_pc[state, action] = (1 - self.learning_rate) * self.q_table_pc[state, action] + self.learning_rate * (reward + self.gamma * max(self.q_table_pc[next_state]))

    def learn_pm(self, state, action, reward, next_state, next_action, is_SARSA=True):
        if is_SARSA:
            self.q_table_pm[state, action] = (1 - self.learning_rate) * self.q_table_pm[state, action] + self.learning_rate * (reward + self.gamma * self.q_table_pm[next_state, next_action])
        else:
            self.q_table_pm[state, action] = (1 - self.learning_rate) * self.q_table_pm[state, action] + self.learning_rate * (reward + self.gamma * max(self.q_table_pm[next_state]))