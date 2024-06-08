# Self-Learning GA for Flexible Job-Shop Scheduling Problem 
This is the implementation of the paper [A self-learning genetic algorithm based on reinforcement learning for flexible job-shop scheduling problem](https://www.sciencedirect.com/science/article/abs/pii/S0360835220304885).

We modified the chromosome representation, crossover operator, and mutation operator.

## How to use
- `BRdata` contains ten Brandimarte’s benchmark instances.
- `main.py` is the entry point of the project.
- `env.py` is the FJSP environment containing GA and RL implementation.

One can run the following command to generate result of a instance (e.g. Mk01):
```console
python3 main.py --instance Mk01
```

To conduct experiment and output a CSV file containing best solution of each run (e.g. Mk01_result.csv):
```console
python3 test.py --instance Mk01 --runs 30 --output Mk01_result
```