import numpy as np


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
    
    return np.round(population, 2)

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
    raw_values = np.array([func(ind) for ind in population])
    
    fitness_scores = 1 / (1 + raw_values)
    
    return raw_values, fitness_scores


def random_selection(population, num_parents):
    """
    Perform random selection to choose parents from the population.
    Return a two-dimensional numpy array containing the selected parents.

    Arguments:
    population -- initial population (a two-dimensional numpy array)
    num_parents -- number of parents to select (a natural number)
    """
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False)
    parents = population[selected_indices]
    
    return parents


def proportional_selection(population, fitness_scores, num_parents):
    """
    Perform proportional (roulette wheel) selection to choose parents from the population.
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array (higher is better, from fitness function)
    num_parents -- number of parents to select
    
    Returns:
    selected parents -- 2D numpy array
    """
    total_fitness = np.sum(fitness_scores)
    probabilities = fitness_scores / total_fitness
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]


def rank_based_selection(population, fitness_scores, num_parents):
    """
    Perform rank-based selection to choose parents from the population.
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array (higher is better, from fitness function)
    num_parents -- number of parents to select
    
    Returns:
    selected parents -- 2D numpy array
    """
    sorted_indices = np.argsort(fitness_scores)
    ranks = np.arange(1, len(fitness_scores)+1)[sorted_indices]
    ranks = ranks[::-1]  # highest fitness gets highest rank
    probabilities = ranks / np.sum(ranks)
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]


def tournament_selection(population, fitness_scores, num_parents, tournament_size=3):
    """
    Perform tournament selection.
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array (higher is better)
    num_parents -- number of parents to select
    tournament_size -- number of individuals competing in each tournament
    
    Returns:
    selected parents -- 2D numpy array
    """
    selected_parents = []
    for _ in range(num_parents):
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_scores = fitness_scores[indices]
        winner_index = indices[np.argmax(tournament_scores)]
        selected_parents.append(population[winner_index])
    return np.array(selected_parents)


def truncation_selection(population, fitness_scores, num_parents, t):
    """
    Perform truncation selection.
    
    Arguments:
    population -- 2D numpy array
    fitness_scores -- 1D numpy array (higher is better)
    num_parents -- number of parents to select
    t -- top t percent of population to consider (0-100)
    
    Returns:
    selected parents -- 2D numpy array
    """
    if t <= 0 or t > 100:
        raise ValueError("t must be between 0 and 100 (exclusive).")
    
    sorted_indices = np.argsort(fitness_scores)  # ascending
    top_t_count = int(np.ceil(len(population) * t / 100))
    top_t_indices = sorted_indices[-top_t_count:]  # pick top t% based on fitness
    top_t_population = population[top_t_indices]
    selected_indices = np.random.choice(len(top_t_population), size=num_parents, replace=False)
    return top_t_population[selected_indices]


def simple_crossover(parents, a):
    """
    Perform simple crossover between two parents to produce two children.
    The crossover point is chosen randomly.

    Arguments:
    parents -- a two-dimensional numpy array containing two parents
               (shape: (2, chromosome_length))
    a -- crossover parameter (a float between 0 and 1)

    Returns:
    Two children (two-dimensional numpy array)
    """
    if parents.ndim != 2 or parents.shape[0] != 2:
        raise ValueError("Parents must be a two-dimensional numpy array with shape (2, chromosome_length).")
    
    if not (0 <= a <= 1):
        raise ValueError("Crossover parameter 'a' must be between 0 and 1.")
    
    parent1 = parents[0]
    parent2 = parents[1]
    chromosome_length = len(parent1)
    
    # Randomly choose a crossover point
    crossover_point = np.random.randint(1, chromosome_length)
    
    # Initialize children with zeros
    child1 = np.zeros(chromosome_length)
    child2 = np.zeros(chromosome_length)
    
    # Transfer the chosen part from parent1 to child1 and from parent2 to child2
    child1[:crossover_point] = parent1[:crossover_point]
    child2[:crossover_point] = parent2[:crossover_point]
    
    # Calculate the remaining part for both children
    for i in range(crossover_point, chromosome_length):
        child1[i] = a * (parent1[i] + parent2[i])
        child2[i] = (1 - a) * (parent1[i] + parent2[i])
    
    return np.array([child1, child2])
