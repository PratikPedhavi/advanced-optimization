from typing import List

import numpy as np


def population_init(population_size: int, data_type: str) -> List[np.ndarray]:
    """ Returns a list of numpy arrays representing the population."""
    if data_type == 'int':
        population = [np.random.randint(low=1, high=10) for _ in range(population_size)]
    else:
        population = [np.random.uniform(low=1, high=10) for _ in range(population_size)]
    return population


def fitness_fitness(x):
    fitness = x^3 + 5 * x^2
    return fitness
