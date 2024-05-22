import numpy as np
import random

from agent import Agent

class SLGAEnv:
    def __init__(self, table_pd, num_jobs, num_machines, dimension, population_size, num_generations, pc = 0.7, pm = 0.01, num_states=20, num_actions=10):
        self.table_pd = table_pd
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        # the length of operation sequence or machine assignment, total length of an individual is 2 * dimension
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
        # Operation sequence
        operation_sequence = np.zeros((self.population_size, self.dimension))
        # count the number of identical elements in the clolumn 'job'
        operation_counts = self.table_pd['job'].value_counts()
        # probability to Choose the job that has the greatest number of operations remaining
        p_cmo = 0.8
        # the next operation in a sequnece is choosen either by the priority rule or randomly
        for i in range(self.population_size):
            operation_counts_tmp = operation_counts.deepcopy()
            for j in range(self.dimension):
                if np.random.rand() < p_cmo:
                    job = operation_counts_tmp.idxmax()
                    operation_counts_tmp[job] -= 1
                    operation_sequence[i][j] = job
                else:
                    job = np.random.choice(operation_counts_tmp.index)
                    operation_counts_tmp[job] -= 1
                    operation_sequence[i][j] = job

        # Machine assignment
        machine_assignment = np.zeros((self.population_size, self.dimension))
        # probability to choose the machine with the shortest processing time for the corresponding operations.
        p_hcms = 0.8
        # find the machine with the shortest processing time for each operation.
        columns_to_check = list(range(self.num_machines))
        self.table_pd['min_machine'] = self.table_pd[columns_to_check].idxmin(axis=1)
        # array to record the next operation to be assigned in a job
        next_operation = np.zeros(operation_counts.shape[0])
        # the next assignment in a sequnece is choosen either by the priority rule or randomly
        for i in range(self.population_size):
            for j in range(self.dimension):
                job = operation_sequence[i][j]
                if np.random.rand() < p_hcms:
                    # find the machine with the shortest processing time for the corresponding operations
                    min_machine = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])]['min_machine'].values[0]
                    machine_assignment[i][j] = min_machine
                else:
                    # randomly choose a machine that is not np.inf in table_pd
                    while(True) :
                        machine = np.random.choice(range(self.num_machines))
                        if self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])][machine].values[0] != np.inf:
                            break
                    machine_assignment[i][j] = machine
                next_operation[job] += 1
        
        return np.hstack([operation_sequence, machine_assignment])

    def fitness(self):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            # record the total processing time of each machine
            time_machine = [0 for _ in range(self.num_machines)]
            # record the time of the last finished operation of each job
            time_job = [0 for _ in range(self.num_jobs)]
            # record the next operation to be assigned in a job
            next_operation = np.zeros(self.num_jobs)
            
            for gene in range(self.dimension):
                job = self.population[i][gene]
                machine = self.population[i][gene + self.dimension]
                processing_time = self.table_pd[(self.table_pd['job'] == job) & (self.table_pd['operation'] == next_operation[job])][str(machine)].values[0]

                if time_job[job] > time_machine[machine]:
                    time_machine[machine] = time_job[job] + processing_time
                    time_job[job] += processing_time
                else:
                    time_machine[machine] += processing_time
                    time_job[job] = time_machine[machine]

                next_operation[job] += 1
        
            makespan = np.max(time_machine)
            fitness[i] = makespan

        return fitness

    def select(self):
        X_new = np.zeros_like(self.population)
        tournamant_size = 3
        for i in range(self.population_size):
            mask = np.random.choice(self.population, size=tournamant_size, replace=True)
            fitness_selected = self.fitness[mask]
            candidates = self.population[mask]
            best_idx = fitness_selected.argmin()
            X_new[i] = candidates[best_idx]

        return X_new
    
    def crossover(self):
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.pc:
                # Select two parents
                parent1 = self.population[i]
                parent2 = self.population[i + 1]
                # randomly split jobs into two exclusive sets
                machine_list = list(range(self.num_jobs))
                job_set1 = set(random.sample(machine_list, int(self.num_jobs / 2)))
                job_set2 = set(machine_list) - job_set1
                # Perform precedence preserving order-based crossover
                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)
                # child 1
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
                # child 2
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
                # Update population
                self.population[i] = child1
                self.population[i + 1] = child2

    def mutate(self):
        for i in range(self.population_size):
            child = self.population[i].deepcopy()
            for idx1 in range(self.dimension):
                # swap mutation
                if np.random.rand() <= self.pm:
                    idx2 = np.random.choice(np.delete(np.arange(self.dimension), idx1))
                    child[idx1], child[idx2] = self.population[idx2], self.population[idx1]
                    child[idx1 + self.dimension], child[idx2 + self.dimension] = self.population[idx2 + self.dimension], self.population[idx1 + self.dimension]
            # Update population
            self.population[i] = child
