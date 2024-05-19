import numpy as np
import random

from agent import Agent

class SLGAEnv:
    def __init__(self, dimension, population_size, num_generations, pc = 0.7, pm = 0.01, num_states=20, num_actions=10):
        self.dimension = dimension
        self.population_size = population_size
        self.num_generations = num_generations
        self.pc = pc
        self.pm = pm
        self.num_states = num_states
        self.num_actions = num_actions
        # 20 states
        self.state_set = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
        # 10 actions each
        self.action_set_pc = [(0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9)]
        self.action_set_pm = [(0.01, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.09), (0.09, 0.11), (0.11, 0.13), (0.13, 0.15), (0.15, 0.17), (0.17, 0.19), (0.19, 0.21)]

        self.population = None
        self.fitness_score = None
        self.first_gen_fitness_score = None

    def reset(self):
        None
    
    def next_state(self):
        w1 = 0.35
        w2 = 0.35
        w3 = 0.3

        f = sum(self.fitness_score) / sum(self.first_gen_fitness_score)
        mean = sum(self.fitness_score) / self.population_size
        mean_first_gen = sum(self.first_gen_fitness_score) / self.population_size
        d = sum(abs(fitness - mean) for fitness in self.fitness_score) / sum(abs(fitness - mean_first_gen) for fitness in self.first_gen_fitness_score)
        m = max(self.fitness_score) / max(self.first_gen_fitness_score)

        s = w1 * f + w2 * d + w3 * m
        state_idx = int(s / 0.05)
    
        # Handle the edge case where s is exactly 1.0
        if s == 1:
            state_idx = self.num_states - 1

        return state_idx

    ''' Major loop of SLGA '''
    def runner(self):
        self.agent = Agent(num_states=20, num_actions=10, epsilon=0.85, learning_rate=0.75, gamma=0.2)

        self.population = self.init_population()
        self.fitness_score = self.fitness()
        self.first_gen_fitness_score = self.fitness_score
        self.best_fitness = max(self.fitness_score)
        self.prev_fitness_score = self.fitness_score
        
        state = random.randint(0, self.num_states - 1)
        action_pc = random.randint(0, self.num_actions - 1)
        action_pm = random.randint(0, self.num_actions - 1)
        t = 0

        for gen in range(self.num_generations):
            rc = (max(self.fitness_score) - self.best_fitness) / self.best_fitness
            rm = (sum(self.fitness_score) - sum(self.prev_fitness_score)) / sum(self.prev_fitness_score)

            next_state = self.next_state()

            if gen < (self.num_states * self.num_actions) / 2:
                # SARSA
                next_action_pc = self.agent.select_action_pc(state)
                next_action_pm = self.agent.select_action_pm(state)
                self.agent.learn_pc(state, action_pc, rc, next_state, next_action_pc, is_SARSA=True)
                self.agent.learn_pm(state, action_pm, rm, next_state, next_action_pm, is_SARSA=True)
            else: 
                # Q-learning
                self.agent.learn_pc(state, action_pc, rc, next_state, None, is_SARSA=False)
                self.agent.learn_pm(state, action_pm, rm, next_state, None, is_SARSA=False)
                next_action_pc = self.agent.select_action_pc(state)
                next_action_pm = self.agent.select_action_pm(state)

            self.best_fitness = max(self.fitness_score)
            self.prev_fitness_score = self.fitness_score
            
            state = next_state
            action_pc = next_action_pc
            action_pm = next_action_pm
            # Execute action
            self.pc = random.uniform(self.action_set_pc[action_pc][0], self.action_set_pc[action_pc][1])
            self.pm = random.uniform(self.action_set_pm[action_pm][0], self.action_set_pm[action_pm][1])
            # Genetic operation
            self.select()
            self.crossover()
            self.mutate()

            t = t + 1
            # Fitness calculation
            self.fitness_score = self.fitness()

    def init_population(self):
        None

    def fitness(self):
        None

    def select(self):
        None
    
    def crossover(self):
        None

    def mutate(self):
        None
