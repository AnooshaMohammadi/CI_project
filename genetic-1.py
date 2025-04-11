import random
import numpy as np

#creating initial binery population
def initial_binary_population(pop_size, chromosome_length):
    return np.random.randint(0, 2, size=(pop_size, chromosome_length))

#creating initial population with real numbers
def initial_real_population(pop_size, chromosome_length, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size=(pop_size, chromosome_length))

