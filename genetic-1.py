import random
import numpy as np

#creating initial binery population
def initial_binary_population(pop_size, chromosome_length):
    return np.random.randint(0, 2, size=(pop_size, chromosome_length))

#creating initial population with real numbers
def initial_real_population(pop_size, chromosome_length, lower_bound, upper_bound):
    population = np.random.uniform(lower_bound, upper_bound, size=(pop_size, chromosome_length))
    return np.round(population, 2)


def proportional_selection(population, fitness_values, num_parents):

    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness

    # Choose parents based on probability distribution
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]
