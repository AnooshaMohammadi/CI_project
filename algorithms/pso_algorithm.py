import numpy as np

def initialize_positions(pop_size, dimension, lower_bound, upper_bound):
    """
    Initialize particle positions randomly within bounds.
    Returns a (pop_size x dimension) numpy array.
    """
    positions = np.random.uniform(lower_bound, upper_bound, size=(pop_size, dimension))
    return np.round(positions, 2)

def initialize_velocities(pop_size, dimension, velocity_bound):
    """
    Initialize particle velocities randomly within [-velocity_bound, velocity_bound].
    Returns a (pop_size x dimension) numpy array.
    """
    velocities = np.random.uniform(-velocity_bound, velocity_bound, size=(pop_size, dimension))
    return np.round(velocities, 2)

def fitness(population, func: callable):
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
    fitness_scores = np.array([func(ind) for ind in population])
    
    return fitness_scores