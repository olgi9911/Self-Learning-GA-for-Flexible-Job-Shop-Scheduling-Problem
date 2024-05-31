import numpy as np
import pandas as pd
import argparse

from env import SLGAEnv

def read_file(path):
    raw_table = pd.read_table(path, header=None)
    #print(raw_table)
    num_jobs = int(raw_table.loc[0, 0])
    num_machines = int(raw_table.loc[0, 1])
    machines_per_operation = int(raw_table.loc[0, 2])
    table = raw_table.loc[1:, 0].str.split(expand=True).astype(float) # Parse each job string
    
    total_operations = int(table[0].sum())
    operation_matrix = []
    for job in range(num_jobs):
        num_operation_list = table[0].astype(int).tolist()
        for operation in range(num_operation_list[job]):
            operation_matrix.append([job, operation])
    operation_matrix = np.array(operation_matrix)
    table.drop([0], axis=1, inplace=True) # Drop the num_operation column
    
    operation_machine_list = []
    for idx, row in table.iterrows():
        row.dropna(inplace=True) # Drop missing values
        row = row.astype(int).tolist()
        while row:
            num_machines_available = row[0] # Available machines for each operation
            operation_machine_list.append(row[1: 1 + num_machines_available * 2])
            row = row[1 + num_machines_available * 2:]

    processing_time_matrix = np.full((total_operations, num_machines), np.inf)
    for operation, row in enumerate(operation_machine_list):
        row = np.array(row).reshape(-1, 2)
        machine = row[:, 0] - 1
        processing_time = row[:, 1]
        processing_time_matrix[operation, machine] = processing_time

    table = np.hstack([processing_time_matrix, operation_matrix])
    table_pd = pd.DataFrame(data=table)
    table_pd.rename(columns={table_pd.columns[-2]: 'job',
                             table_pd.columns[-1]: 'operation'},
                             inplace=True)
    table_pd['job'] = table_pd['job'].astype(int)
    table_pd['operation'] = table_pd['operation'].astype(int)
    table_pd['O'] = table_pd['job'].astype(str) + table_pd['operation'].astype(str)

    #print(table_pd)
    return table_pd, num_jobs, num_machines, total_operations


parser = argparse.ArgumentParser(description="config")
parser.add_argument("--instance", type=str, default="Mk01")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--runs", type=int, default=1)
args = parser.parse_args()

num_runs = args.runs
solutions = np.zeros((num_runs, 2))

data, num_jobs, num_machines, total_operations = read_file(f"Brdata/{args.instance}.fjs")
for i in range(num_runs):
    env = SLGAEnv(data, num_jobs, num_machines, dimension=total_operations, population_size= 5 * num_jobs * num_machines, num_generations= 5 * num_jobs * num_machines)
    solutions[i] = env.runner()

table = pd.DataFrame(solutions, columns=['Best', 'Generation'])
table.to_csv(f"{args.output}.csv")