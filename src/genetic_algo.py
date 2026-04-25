from typing import List

import numpy as np


def population_init(population_size: int, gene_count: int, data_type: str) -> List[np.ndarray]:
    """ Returns a list of numpy arrays representing the population."""
    if data_type == 'int':
        population_arr = [np.random.randint(low=1, high=10, size=gene_count) for _ in range(population_size)]
    else:
        population_arr = [np.random.randint(low=1, high=10, size=gene_count) for _ in range(population_size)]
    return population_arr


def fitness_function(x):
    fitness_array = x**3 + 5 * x**2
    return fitness_array


def parent_selection(population: List[np.ndarray]) -> List[np.ndarray]:
    selected_idx = np.random.choice(range(len(population)), size=2, replace=False)
    return [population[selected_idx[0]], population[selected_idx[1]]]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


if __name__ == '__main__':
    population_size = 10
    gene_count = 4
    population = population_init(population_size, gene_count, 'int')
    fitness = list(map(fitness_function, population))
    parents = parent_selection(population)
    child = crossover(parents[0], parents[1])
    print(population)
    print(fitness)
    print("Parents: {}".format(parents))
    print("Child: {}".format(child))