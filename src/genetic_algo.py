from typing import List

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class Individual:
    gene_sequence: list
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.gene_sequence)


@dataclass
class Population:
    individuals: List[Individual]
    size: int


class GeneticAlgorithm:
    def __init__(self, population_size, gene_count):
        self.best_fitness = []
        self.termination_criteria = 0
        self.population = self.population_init(population_size, gene_count, 'int')

    def optimize(self):
        while self.termination_criteria < 400:
            fitness = list(map(self.fitness_function, self.population))
            self.best_fitness.append(max(fitness))
            parents, selected_idx = self.parent_selection(self.population, 'tournament')
            child = self.crossover(parents[0], parents[1])
            child_new = self.mutation(child)
            if np.min(fitness) <= self.fitness_function(child_new):
                self.population[np.argmin(fitness)] = child_new
            self.termination_criteria += 1

        print("First Fitness Values: {}".format(self.best_fitness[:5]))
        print("Best Fitness Values: {}".format(self.best_fitness[-5:]))
        self.plot_fitness(self.best_fitness)

    def population_init(self, population_size: int, gene_count: int, data_type: str) -> List[np.ndarray]:
        """ Returns a list of numpy arrays representing the population."""
        population_arr = list()
        np.random.seed(42)
        for _ in range(population_size):
            individual = Individual(np.random.randint(low=1, high=10, size=gene_count))
            population_arr.append(individual)
        print("Initial Population: {}".format(population_arr))
        return population_arr


    def fitness_function(self, individual: Individual):
        """ Returns the fitness function for given individual."""
        fitness_array = individual.gene_sequence**3 + 5 * individual.gene_sequence**2
        return np.sum(fitness_array)


    def parent_selection(self, population: List[np.ndarray], selection_method: str = 'roulette') -> (List[np.ndarray], int):
        """ Returns a list of numpy arrays representing the population.
            For selection method 'random', the population will be randomly selected.
            For selection method 'roulette', the population will be selected based on selection probabilities
            that are calculated based on individual fitness.
        """
        if selection_method == 'roulette':
            fitness_values = np.array(list(map(self.fitness_function, population)))
            total_fitness = np.sum(fitness_values)
            selection_probabilities = fitness_values / total_fitness
            selected_idx = np.random.choice(range(len(population)), size=2, replace=False, p=selection_probabilities)
        elif selection_method == 'tournament':
            tournament_size = round(len(population) / 2)
            tournament_idx = np.random.choice(range(len(population)), size=tournament_size, replace=False)
            selected_idx = np.random.choice(tournament_idx, size=2, replace=False)
        else: #if selection_method == 'random':
            selected_idx = np.random.choice(range(len(population)), size=2, replace=False)
        return [population[selected_idx[0]], population[selected_idx[1]]], selected_idx


    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """ Returns a crossover between two parents."""
        crossover_point = np.random.randint(1, parent1.size - 1)
        child_sequence = np.concatenate((parent1.gene_sequence[:crossover_point],
                                         parent2.gene_sequence[crossover_point:]))
        return Individual(child_sequence)


    def mutation(self, child: Individual, scramble_len: int = 4) -> Individual:
        """ Returns the child array after performing a mutation operation (scramble)."""
        child_sequence = np.copy(child.gene_sequence)
        start_idx = np.random.randint(0, len(child_sequence)-scramble_len)
        segment = child_sequence[start_idx:start_idx + scramble_len].copy()
        rng = np.random.default_rng()
        array_new = rng.permutation(segment)
        child_sequence[start_idx:start_idx + scramble_len] = array_new
        return Individual(child_sequence)


    def plot_fitness(self, fitness_tracker):
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
    GeneticAlgorithm(population_size, gene_count).optimize()
