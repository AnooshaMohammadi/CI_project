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
    
    fitness_scores = np.max(raw_values) - raw_values + 1e-6
    
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

#######################################
######-Real population crossover-######
#######################################

def simple_crossover(parents, a=0.5, crossover_rate=0.75):
    """
    Perform simple crossover on an array of parents (real-valued population).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    a -- crossover parameter (0 <= a <= 1)
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        raise ValueError("Number of parents must be even for pairing.")

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        if np.random.rand() < crossover_rate:
            # Random crossover point
            crossover_point = np.random.randint(1, chromosome_length)
            
            child1 = np.zeros(chromosome_length)
            child2 = np.zeros(chromosome_length)
            
            child1[:crossover_point] = parent1[:crossover_point]
            child2[:crossover_point] = parent2[:crossover_point]

            for j in range(crossover_point, chromosome_length):
                child1[j] = a * (parent1[j] + parent2[j])
                child2[j] = (1 - a) * (parent1[j] + parent2[j])
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        children.append(child1)
        children.append(child2)

    return np.round(children, 1)


def simple_arithmetic_crossover(parents, alpha=0.5, crossover_rate=0.75):
    """
    Perform Simple Arithmetic Crossover.
    Only one gene per pair is modified using:
    - Child1[pos] = a * (parent1[pos] + parent2[pos])
    - Child2[pos] = (1 - a) * (parent1[pos] + parent2[pos])

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    alpha -- crossover parameter (0 <= alpha <= 1), scaling factor
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        raise ValueError("Number of parents must be even for pairing.")

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        if np.random.rand() < crossover_rate:
            pos = np.random.randint(0, chromosome_length)

            child1 = parent1.copy()
            child2 = parent2.copy()

            sum_val = parent1[pos] + parent2[pos]
            child1[pos] = alpha * sum_val
            child2[pos] = (1 - alpha) * sum_val
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.round(children, 1)


def whole_arithmetic_crossover(parents, alpha=0.5, crossover_rate=0.75):
    """
    Perform Whole Arithmetic Crossover.
    All genes are modified using:
    - Child1[i] = a * (parent1[i] + parent2[i])
    - Child2[i] = (1 - a) * (parent1[i] + parent2[i])

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    alpha -- crossover parameter (0 <= alpha <= 1), scaling factor
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        raise ValueError("Number of parents must be even for pairing.")

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        if np.random.rand() < crossover_rate:
            sum_vals = parent1 + parent2
            child1 = alpha * sum_vals
            child2 = (1 - alpha) * sum_vals
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.round(children, 1)

#######################################
######-Real population mutation-#######
#######################################

def complement_mutation(population, lower_bound=0, upper_bound=1, mutation_rate=0.01):
    """
    Perform complement mutation on a real-valued population.
    For each individual, randomly select a gene and replace it with (upper bound + lower_bound - value).
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    mutation_rate -- probability of mutating each gene (float between 0 and 1)
    
    Returns:
    mutated_population -- 2D numpy array after mutation
    """
    mutated = population.copy()
    num_individuals, chromosome_length = mutated.shape
    
    for i in range(num_individuals):
        for j in range(chromosome_length):
            if np.random.rand() < mutation_rate:
                mutated[i, j] = upper_bound + lower_bound - mutated[i, j]
    
    return np.round(mutated, 1)

#######################################
######-permutation population mutation-#######
#######################################


def swap_mutation(population, mutation_rate=0.01):
    """
    Perform swap mutation on a permutation-based population.
    Randomly selects two positions in a chromosome and swaps their values.
    
    Arguments:
    population -- 2D numpy array (each row is a permutation)
    mutation_rate -- probability of performing mutation per individual
    
    Returns:
    mutated_population -- 2D numpy array after mutation
    """
    mutated = population.copy()
    num_individuals, chromosome_length = mutated.shape
    
    for i in range(num_individuals):
        if np.random.rand() < mutation_rate:
            pos1, pos2 = np.random.choice(chromosome_length, size=2, replace=False)
            mutated[i, pos1], mutated[i, pos2] = mutated[i, pos2], mutated[i, pos1]
    
    return mutated


def insert_mutation(population, mutation_rate=0.01):
    """
    Perform insert mutation on a permutation-based population.
    Randomly selects a gene and inserts it into another position.
    
    Arguments:
    population -- 2D numpy array (each row is a permutation)
    mutation_rate -- probability of performing mutation per individual
    
    Returns:
    mutated_population -- 2D numpy array after mutation
    """
    mutated = population.copy()
    num_individuals, chromosome_length = mutated.shape
    
    for i in range(num_individuals):
        if np.random.rand() < mutation_rate:
            # Pick two distinct positions
            pos1, pos2 = np.sort(np.random.choice(chromosome_length, size=2, replace=False))
            
            # Always move the gene at the higher index (pos2) 
            # to just after the lower index (pos1)
            value = mutated[i, pos2]
            
            # Delete the gene at pos2
            temp = np.delete(mutated[i], pos2)
            
            # Insert it just after pos1
            mutated[i] = np.insert(temp, pos1 + 1, value)
    
    return mutated