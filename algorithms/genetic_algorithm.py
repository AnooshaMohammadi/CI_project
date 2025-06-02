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


def fitness(population, problem_type):
    """
    Evaluate initial population's fitness.
    Return a numpy array.

    Arguments:
    population -- initial population (a two-dimensional numpy array)
    problem_type -- either min or max type

    Type of fitness function: sum of squares. 
    """
    if problem_type == "max":
        fitness_values = np.sum(np.square(population), axis=1)
    elif problem_type == "min":
        fitness_values = 1 / (1 + np.sum(np.square(population), axis=1))
    
    return fitness_values


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


def proportional_selection(population, fitness_values, num_parents):
    """
    Perform proportional selection to choose parents from the population.
    Return a two-dimensional numpy array containing the selected parents.

    Arguments:
    population -- initial population (a two-dimensional numpy array)
    fitness_values -- fitness values of each individual in the population (a one-dimensional numpy array)
    num_parents -- number of parents to select (a natural number)
    """
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    # Choose parents based on probability distribution
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    
    return population[selected_indices]


def rank_based_selection(population, fitness_values, num_parents, problem_type):
    """
    Perform rank-based selection to choose parents from the population.
    Return a two-dimensional numpy array containing the selected parents.

    Arguments:
    population -- initial population (a two-dimensional numpy array)
    fitness_values -- fitness values of each individual in the population (a one-dimensional numpy array)
    num_parents -- number of parents to select (a natural number)
    problem_type -- either min or max type
    """
    # Sort the fitness values and get the sorted indices
    sorted_indices = np.argsort(fitness_values)
    
    # Assign ranks (lower rank means better fitness for maximization, higher rank for minimization)
    if problem_type == "max":  # Assuming maximization problem
        ranks = np.arange(1, len(fitness_values) + 1)[sorted_indices]
    elif problem_type == "min":  # Assuming minimization problem
        ranks = np.arange(len(fitness_values), 0, -1)[sorted_indices]
    
    # Calculate probabilities based on ranks
    total_rank = np.sum(ranks)
    probabilities = ranks / total_rank
    
    # Choose parents based on probability distribution
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    
    return population[selected_indices]


def tournament_selection(population, fitness_values, num_parents, tournament_size=3):
    selected_parents = []

    for _ in range(num_parents):
        # Randomly pick tournament_size individuals
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament = population[indices]
        tournament_fitness = fitness_values[indices]

        # Select the individual with the highest fitness
        winner_index = indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_index])
    
    return np.array(selected_parents)


def truncation_selection(population, fitness_values, num_parents, t, problem_type):
    """
    Perform truncation selection to choose parents from the population.
    First sort the population by fitness values, then select the top t percent of the population,
    and finally choose n parents from these t percent.

    Return a two-dimensional numpy array containing the selected parents.

    Arguments:
    population -- initial population (a two-dimensional numpy array)
    fitness_values -- fitness values of each individual in the population (a one-dimensional numpy array)
    num_parents -- number of parents to select (a natural number)
    t -- percentage of the population to consider (0 to 100)
    problem_type -- either 'min' or 'max' type
    """
    if t <= 0 or t > 100:
        raise ValueError("t_percent must be between 0 and 100 (exclusive of 0).")
    
    # Sort the fitness values and get the sorted indices
    sorted_indices = np.argsort(fitness_values)
    
    if problem_type == "max":
        # For maximization, select the top t_percent individuals
        top_t_indices = sorted_indices[-int(np.ceil(len(population) * t / 100)):]
    elif problem_type == "min":
        # For minimization, select the bottom t_percent individuals
        top_t_indices = sorted_indices[:int(np.ceil(len(population) * t / 100))]
    else:
        raise ValueError("problem_type must be either 'min' or 'max'")
    print(top_t_indices)
    # Select the top t_percent individuals
    top_t_population = population[top_t_indices]
    print("top_t_population", top_t_population)
    
    # Randomly select num_parents from the top t_percent individuals
    selected_indices = np.random.choice(len(top_t_population), size=num_parents, replace=False)
    parents = top_t_population[selected_indices]
    
    return parents


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
