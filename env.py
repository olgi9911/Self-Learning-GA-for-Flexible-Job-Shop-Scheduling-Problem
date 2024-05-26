import numpy as np
import random

from agent import Agent

class SLGAEnv:
    def __init__(self, table_pd, num_jobs, num_machines, dimension, population_size, num_generations, pc = 0.7, pm = 0.001, num_states=20, num_actions=10):
        self.table_pd = table_pd
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        # The length of operation sequence or machine assignment, total length of an individual is 2 * dimension
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

    def next_state(self):
        w1 = 0.35
        w2 = 0.35
        w3 = 0.3

        f = np.sum(self.fitness_score) / np.sum(self.first_gen_fitness_score)
        mean = np.mean(self.fitness_score)
        mean_first_gen = np.mean(self.first_gen_fitness_score)
        d = np.sum(np.abs(self.fitness_score - mean)) / np.sum(np.abs(self.first_gen_fitness_score - mean_first_gen))
        m = min(self.fitness_score) / min(self.first_gen_fitness_score)
        
        s = w1 * f + w2 * d + w3 * m
        #print(f's = {s}')
        state_idx = min(int(s / 0.05), self.num_states - 1) # Handle the edge case

        return state_idx

    def runner(self):
        ''' Major loop of SLGA '''
        self.agent = Agent(num_states=20, num_actions=10, epsilon=0.85, learning_rate=0.75, gamma=0.2)

        self.population = self.init_population()
        self.fitness_score = self.fitness()
        self.first_gen_fitness_score = self.fitness_score.copy()
        self.best_fitness = np.min(self.fitness_score)
        self.prev_fitness_score = self.fitness_score
        
        state = random.randint(0, self.num_states - 1)
        action_pc = random.randint(0, self.num_actions - 1)
        action_pm = random.randint(0, self.num_actions - 1)

        for gen in range(self.num_generations):
            rc = (np.min(self.fitness_score) - np.min(self.prev_fitness_score)) / np.min(self.prev_fitness_score)
            rm = (np.sum(self.fitness_score) - np.sum(self.prev_fitness_score)) / np.sum(self.prev_fitness_score)

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
            
            state = next_state
            action_pc = next_action_pc
            action_pm = next_action_pm
            # Execute action
            self.pc = random.uniform(self.action_set_pc[action_pc][0], self.action_set_pc[action_pc][1])
            self.pm = random.uniform(self.action_set_pm[action_pm][0], self.action_set_pm[action_pm][1])
            # Genetic operation
            elite_population = self.select()
            self.crossover()
            self.mutate()

            self.prev_fitness_score = self.fitness_score
            # Fitness calculation
            self.fitness_score = self.fitness()

            self.elite_retention(elite_population)
            self.fitness_score = self.fitness()

            current_best_fitness = min(self.fitness_score)
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
            print(f"{gen + 1 : >3}: Best fitness = {self.best_fitness}")

        return self.best_fitness

    def init_population(self):
        operation_sequence = np.zeros((self.population_size, self.dimension), dtype=int)
        operation_counts = self.table_pd['job'].value_counts() # Count the number of identical elements in the column 'job'
        p_cmo = 0.8 # Probability to Choose the job that has the greatest number of operations remaining

        # The next operation in a sequence is chosen either by the priority rule or randomly
        for i in range(self.population_size):
            operation_counts_tmp = (operation_counts).copy()
            for j in range(self.dimension):
                if np.random.rand() < p_cmo:
                    job = operation_counts_tmp.idxmax()
                    operation_counts_tmp[job] -= 1
                    operation_sequence[i][j] = job
                else:
                    # Make sure that the job isn't completely assigned to the sequence
                    while True:
                        job = np.random.choice(operation_counts_tmp.index)
                        if operation_counts_tmp[job] > 0:
                            break
                    operation_counts_tmp[job] -= 1
                    operation_sequence[i][j] = job

        machine_assignment = np.zeros((self.population_size, self.dimension), dtype=int)
        p_hcms = 0.8 # Probability to choose the machine with the shortest processing time for the corresponding operations

        # Find the machine with the shortest processing time for each operation.
        columns_to_check = list(range(self.num_machines))
        self.table_pd['min_machine'] = self.table_pd[columns_to_check].idxmin(axis=1)

        # The next assignment in a sequence is chosen either by the priority rule or randomly
        for i in range(self.population_size):
            next_operation = np.zeros(self.num_jobs, dtype=int) # Array to record the next operation to be assigned in a job
            for j in range(self.dimension):
                job = operation_sequence[i][j]
                if np.random.rand() < p_hcms:
                    # Find the machine with the shortest processing time for the corresponding operations
                    min_machine = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])]['min_machine'].values[0]
                else:
                    # Randomly choose a machine that is not np.inf in table_pd
                    values = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])][columns_to_check].values.flatten()
                    non_inf_indices = [idx for idx, val in enumerate(values) if val != np.inf]
                    min_machine = random.choice(non_inf_indices)
                    
                machine_assignment[i][j] = min_machine
                next_operation[job] += 1
        
        #print('Population initialization finished')
        return np.hstack([operation_sequence, machine_assignment])

    def fitness(self):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            time_machine = [0 for _ in range(self.num_machines)] # Record the total processing time of each machine
            time_job = [0 for _ in range(self.num_jobs)] # Record the time of the last finished operation of each job
            next_operation = np.zeros(self.num_jobs, dtype=int) # Record the next operation to be assigned in a job
            
            for gene in range(self.dimension):
                job = self.population[i][gene]
                machine = self.population[i][gene + self.dimension]
                processing_time = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])][machine].values[0]
                
                start_time = max(time_job[job], time_machine[machine])
                time_machine[machine] = start_time + processing_time
                time_job[job] = time_machine[machine]

                next_operation[job] += 1

            makespan = max(time_machine)
            fitness[i] = makespan

        #print(fitness)
        return fitness

    def select(self):
        elite_size = int(self.population_size * 0.1)
        elite_indices = np.argsort(self.fitness_score)[:elite_size]
        elite_population = self.population[elite_indices]
        
        return elite_population

    def elite_retention(self, elite_population):
        elite_size = elite_population.shape[0]
        worst_indices = np.argsort(self.fitness_score)[-elite_size:]
        self.population[worst_indices] = elite_population
    
    def crossover(self):
        ''' Precedence preserving order-based crossover (POX) '''
        half_num_jobs = self.num_jobs // 2

        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.pc:
                # Select two parents
                parent1 = self.population[i]
                parent2 = self.population[i + 1]

                # Randomly split jobs into two exclusive sets
                job_set1 = set(random.sample(range(self.num_jobs), half_num_jobs))
                job_set2 = set(range(self.num_jobs)) - job_set1

                # Perform precedence preserving order-based crossover
                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)
                # Child 1
                idx_p2 = 0
                for j in range(self.dimension):
                    if parent1[j] in job_set1:
                        child1[j] = parent1[j]
                        child1[j + self.dimension] = parent1[j + self.dimension]
                    else:
                        while parent2[idx_p2] in job_set1:
                            idx_p2 += 1
                        child1[j] = parent2[idx_p2]
                        child1[j + self.dimension] = parent2[idx_p2 + self.dimension]
                        idx_p2 += 1
                # Child 2
                idx_p1 = 0
                for j in range(self.dimension):
                    if parent2[j] in job_set2:
                        child2[j] = parent2[j]
                        child2[j + self.dimension] = parent2[j + self.dimension]
                    else:
                        while parent1[idx_p1] in job_set2:
                            idx_p1 += 1
                        child2[j] = parent1[idx_p1]
                        child2[j + self.dimension] = parent1[idx_p1 + self.dimension]
                        idx_p1 += 1
                    
                # Update population
                self.population[i] = child1
                self.population[i + 1] = child2

    def mutate(self):
        ''' Swap mutation '''
        columns_to_check = list(range(self.num_machines))
        for i in range(self.population_size):
            for idx1 in range(self.dimension):
                if np.random.rand() <= self.pm:
                    job = self.population[i][idx1]
                    operation = np.sum(self.population[i][:idx1] == job)
                    
                    job_ops = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == operation)]
                    values = job_ops[columns_to_check].values.flatten()
                    non_inf_indices = np.where(values != np.inf)[0]
                    machine_new = random.choice(non_inf_indices)

                    self.population[i][idx1 + self.dimension] = machine_new
                    
