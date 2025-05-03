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


#creating initial population with real numbers
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
        fitness_value = np.sum(np.square(population), axis=1)
    elif problem_type == "min":
        fitness_value = 1 / (1 + np.sum(np.square(population), axis=1))
    return fitness_value

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