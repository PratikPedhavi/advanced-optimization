from typing import List

import numpy as np
import matplotlib.pyplot as plt


def population_init(population_size: int, gene_count: int, data_type: str) -> List[np.ndarray]:
    """ Returns a list of numpy arrays representing the population."""
    population_arr = list()
    np.random.seed(42)
    for _ in range(population_size):
        population_arr.append(np.random.randint(low=1, high=10, size=gene_count))
    print("Initial Population: {}".format(population_arr))
    return population_arr


def fitness_function(x):
    fitness_array = x**3 + 5 * x**2
    return np.sum(fitness_array)


def parent_selection(population: List[np.ndarray], selection_method: str = 'roulette') -> (List[np.ndarray], int):
    if selection_method == 'random':
        selected_idx = np.random.choice(range(len(population)), size=2, replace=False)
    else: # selection_method == 'roulette':
        fitness_values = np.array(list(map(fitness_function, population)))
        total_fitness = np.sum(fitness_values)
        selection_probabilities = fitness_values / total_fitness
        selected_idx = np.random.choice(range(len(population)), size=2, replace=False, p=selection_probabilities)
    return [population[selected_idx[0]], population[selected_idx[1]]], selected_idx


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


def mutation(child_array: np.ndarray, scramble_len: int = 4) -> np.ndarray:
    scramble_point = np.random.randint(0, len(child_array)-scramble_len)
    scramble_array = child_array[scramble_point:scramble_point + scramble_len].copy()
    rng = np.random.default_rng()
    array_new = rng.permutation(scramble_array)
    child_array[scramble_point:scramble_point + scramble_len] = array_new
    return child_array


def plot_fitness(fitness_tracker):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1,len(fitness_tracker)+1), fitness_tracker, marker="o", color="blue", linestyle="-")
    ax.set_title("Fitness Evolution")
    ax.set_xlabel("Steps counter")
    ax.set_ylabel("Fitness Value")
    plt.grid(True)
    plt.show()
    return


if __name__ == '__main__':
    population_size = 10
    gene_count = 8
    best_fitness = []
    population = population_init(population_size, gene_count, 'int')
    termination_criteria = 0
    while termination_criteria < 400:
        fitness = list(map(fitness_function, population))
        best_fitness.append(max(fitness))
        least_fitness = min(fitness)
        parents, selected_idx = parent_selection(population)
        child = crossover(parents[0], parents[1])
        child_new = mutation(child.copy())
        if np.min(fitness) <= fitness_function(child_new):
            population[np.argmin(fitness)] = child_new
        termination_criteria+=1

    print("First Fitness Values: {}".format(best_fitness[:5]))
    print("Best Fitness Values: {}".format(best_fitness[-5:]))
    plot_fitness(best_fitness)

