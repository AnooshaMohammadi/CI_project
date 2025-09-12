import numpy as np
import benchmarkfcns as bf

def initial_binary_population(pop_size, chromosome_length):
    """
    Create initial binery population.
    Return a numpy array.

    Arguments:
    pop_size -- population size (a natural number)
    chromosome_length -- number of genes in each chromosome (a natural number)
    """
    return np.random.randint(0, 2, size=(pop_size, chromosome_length))


def initial_real_population(pop_size, chromosome_length, lower_bound, upper_bound):
    """
    Create initial real population.
    Return a two-dimensional numpy array.
    
    Arguments:
    pop_size -- population size (a natural number)
    chromosome_length -- number of genes in each chromosome (a natural number)
    lower_bound, upper_bound -- two arguments that describe an interval for the boundaries of the population
    """
    population = np.random.uniform(lower_bound, upper_bound, size=(pop_size, chromosome_length))
    
    return np.round(population, 1)

def initial_permutation_population(pop_size, chromosome_length):
    """
    Create initial population where each chromosome is a random permutation.
    Return a two-dimensional numpy array.
    
    Arguments:
    pop_size -- population size (a natural number)
    chromosome_length -- number of genes in each chromosome (a natural number)
    """
    population = np.zeros((pop_size, chromosome_length), dtype=int)
    for i in range(pop_size):
        population[i] = np.random.permutation(chromosome_length)
    return population


def fitness(population, func):
    """
    Evaluate a population on a given benchmark function (minimization).
    
    Arguments:
    population -- 2D numpy array (each row is an individual solution)
    func       -- callable benchmark function that accepts a 1D array (individual)
    
    Returns:
    raw_values     -- 1D numpy array of raw function values (lower is better)
    fitness_scores -- 1D numpy array of converted scores (higher is better), 
                      used for selection operators
    """
    fitness_func = getattr(bf, func)
    fitness_scores = fitness_func(population)
    
    return fitness_scores
