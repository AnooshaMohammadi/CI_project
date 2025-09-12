import numpy as np

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


def proportional_selection(population, fitness_scores, num_parents, problem_type="min"):
    """
    Perform proportional (roulette wheel) selection.
    Works for both minimization and maximization problems,
    even when fitness values are negative.
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array (raw objective values)
    num_parents -- number of parents to select
    problem_type -- "min" or "max"
    
    Returns:
    selected parents -- 2D numpy array
    """
    if problem_type == "min":
        # shift so all values are positive
        adj_fitness = np.max(fitness_scores) - fitness_scores + 1e-8
    else:
        adj_fitness = fitness_scores - np.min(fitness_scores) + 1e-8  # ensure non-negative

    probabilities = adj_fitness / np.sum(adj_fitness)
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]


def rank_based_selection(population, fitness_scores, num_parents, problem_type="min"):
    """
    Perform rank-based selection to choose parents from the population.
    Works for both maximization and minimization problems.
    
    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array of fitness values
    num_parents -- number of parents to select
    problem_type -- "max" for maximization, "min" for minimization
    
    Returns:
    selected parents -- 2D numpy array
    """

    if problem_type == "max":
        sorted_indices = np.argsort(fitness_scores)  # ascending
    else:
        sorted_indices = np.argsort(-fitness_scores)  # descending
    # Assign ranks: highest rank for best fitness
    ranks = np.arange(1, len(fitness_scores) + 1)[sorted_indices]
    ranks = ranks[::-1]

    probabilities = ranks / np.sum(ranks)
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]


def tournament_selection(population, fitness_scores, num_parents, problem_type="min"):
    """
    Perform tournament selection.
    Works for both maximization and minimization problems.

    Arguments:
    population -- 2D numpy array (each row is an individual)
    fitness_scores -- 1D numpy array of fitness values
    num_parents -- number of parents to select
    problem_type -- "max" for maximization, "min" for minimization

    Returns:
    selected parents -- 2D numpy array
    """
    tournament_size = max(2, int(num_parents * 1.5))  # ensure at least 2
    selected_parents = []

    for _ in range(num_parents):
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_scores = fitness_scores[indices]

        if problem_type == "max":
            winner_index = indices[np.argmax(tournament_scores)]
        else:
            winner_index = indices[np.argmin(tournament_scores)]

        selected_parents.append(population[winner_index])

    return np.array(selected_parents)


def truncation_selection(population, fitness_scores, num_parents, t=75, problem_type="min"):
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
    """
    Perform truncation selection.
    Works for both maximization and minimization problems.

    Arguments:
    population -- 2D numpy array
    fitness_scores -- 1D numpy array
    num_parents -- number of parents to select
    t -- top t percent of population to consider (0-100)
    problem_type -- "max" for maximization, "min" for minimization

    Returns:
    selected parents -- 2D numpy array
    """
    if t <= 0 or t > 100:
        raise ValueError("t must be between 0 and 100 (exclusive).")
    
    if problem_type == "max":
        sorted_indices = np.argsort(fitness_scores)  # ascending
        top_t_count = int(np.ceil(len(population) * t / 100))
        top_t_indices = sorted_indices[-top_t_count:]  # pick top t% for max problem
    else:  # min problem
        sorted_indices = np.argsort(fitness_scores)  # ascending
        top_t_count = int(np.ceil(len(population) * t / 100))
        top_t_indices = sorted_indices[:top_t_count]  # pick top t% for min problem

    top_t_population = population[top_t_indices]
    selected_indices = np.random.choice(len(top_t_population), size=num_parents, replace=False)
    return top_t_population[selected_indices]

#######################################
######-Binery population crossover-######
#######################################

def one_point_crossover(parents, crossover_rate=0.75):
    """
    Perform one-point crossover on an array of parents (real-valued population).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        num_parents = num_parents - 1

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        if np.random.rand() < crossover_rate:
            # pick a crossover point (not 0, not end)
            point = np.random.randint(1, chromosome_length)

            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            # no crossover → copy parents
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.array(children)


def two_point_crossover(parents, crossover_rate=0.75):
    """
    Perform two-point crossover on an array of parents (real-valued population).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
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
            # pick two distinct crossover points
            p1, p2 = sorted(np.random.choice(range(1, chromosome_length), 2, replace=False))

            child1 = np.concatenate((parent1[:p1], parent2[p1:p2], parent1[p2:]))
            child2 = np.concatenate((parent2[:p1], parent1[p1:p2], parent2[p2:]))
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.array(children)


def uniform_crossover(parents, crossover_rate=0.75, swap_prob=0.5):
    """
    Perform uniform crossover on an array of parents (real-valued population).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    crossover_rate -- probability of performing crossover for each pair
    swap_prob -- probability of swapping each gene

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
            mask = np.random.rand(chromosome_length) < swap_prob
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.array(children)

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
        num_parents = num_parents - 1

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


def simple_arithmetic_crossover(parents, a=0.5, crossover_rate=0.75):
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
        num_parents = num_parents - 1

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        if np.random.rand() < crossover_rate:
            pos = np.random.randint(0, chromosome_length)

            child1 = parent1.copy()
            child2 = parent2.copy()

            sum_val = parent1[pos] + parent2[pos]
            child1[pos] = a * sum_val
            child2[pos] = (1 - a) * sum_val
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.round(children, 1)


def whole_arithmetic_crossover(parents, a=0.5, crossover_rate=0.75):
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
        num_parents = num_parents - 1

    children = []

    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        if np.random.rand() < crossover_rate:
            sum_vals = parent1 + parent2
            child1 = a * sum_vals
            child2 = (1 - a) * sum_vals
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.round(children, 1)

#######################################
######-permutation population crossover-######
#######################################

def order_crossover(parents, crossover_rate=0.75):
    """
    Perform two-point Order Crossover (OX) on a population of parents (permutation-based).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        raise ValueError("Number of parents must be even for pairing.")

    children = []

    for i in range(0, num_parents, 2):
        p1 = parents[i]
        p2 = parents[i+1]

        if np.random.rand() < crossover_rate:
            # choose two random crossover points
            start, end = sorted(np.random.choice(range(chromosome_length), 2, replace=False))
        
            # initialize children
            c1 = np.full(chromosome_length, -1)
            c2 = np.full(chromosome_length, -1)

            # copy slices
            c1[start:end] = p1[start:end]
            c2[start:end] = p2[start:end]

            def fill(child, donor, start, end):
                pos = end % chromosome_length
                for j in range(chromosome_length):
                    gene = donor[(end + j) % chromosome_length]
                    if gene not in child:
                        child[pos] = gene
                        pos = (pos + 1) % chromosome_length
                return child

            c1 = fill(c1, p2, start, end)
            c2 = fill(c2, p1, start, end)
        else:
            # no crossover, just copy
            c1 = p1.copy()
            c2 = p2.copy()

        children.append(c1)
        children.append(c2)

    return np.array(children)


def cycle_crossover_batch(parents, crossover_rate=0.75):
    """
    Perform Cycle Crossover (CX) on an array of parents (permutation population).

    Arguments:
    parents -- 2D numpy array of selected parents (shape: even_number x chromosome_length)
    crossover_rate -- probability of performing crossover for each pair

    Returns:
    children -- 2D numpy array of offspring (same shape as parents)
    """
    num_parents, chromosome_length = parents.shape

    if num_parents % 2 != 0:
        raise ValueError("Number of parents must be even for pairing.")

    children = []

    for i in range(0, num_parents, 2):
        parent1, parent2 = parents[i], parents[i+1]

        if np.random.rand() < crossover_rate:
            child1 = [-1] * chromosome_length
            child2 = [-1] * chromosome_length

            # Track cycles
            visited = [False] * chromosome_length
            cycle = 0

            for start in range(chromosome_length):
                if not visited[start]:
                    idx = start
                    while not visited[idx]:
                        visited[idx] = True
                        if cycle % 2 == 0:  # even cycle → copy parent1 to child1, parent2 to child2
                            child1[idx] = parent1[idx]
                            child2[idx] = parent2[idx]
                        else:  # odd cycle → swap
                            child1[idx] = parent2[idx]
                            child2[idx] = parent1[idx]
                        idx = np.where(parent1 == parent2[idx])[0][0]
                    cycle += 1
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        children.append(child1)
        children.append(child2)

    return np.array(children)

#######################################
######-binery population mutation-#######
#######################################

def bit_flip_mutation(population, mutation_rate=0.01):
    """
    Perform bit-flip mutation on a binary population.
    Each gene has a probability of being flipped (0 -> 1, 1 -> 0).

    Arguments:
    population -- 2D numpy array (each row is an individual, genes are 0/1)
    mutation_rate -- probability of mutating each gene (float between 0 and 1)

    Returns:
    mutated_population -- 2D numpy array after mutation
    """
    mutated = population.copy()
    num_individuals, chromosome_length = mutated.shape

    for i in range(num_individuals):
        for j in range(chromosome_length):
            if np.random.rand() < mutation_rate:
                mutated[i, j] = 1 - mutated[i, j]  # flip 0<->1

    return mutated

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
            pos1, pos2 = np.sort(np.random.choice(chromosome_length, size=2, replace=False))
            value = mutated[i, pos2]
            temp = np.delete(mutated[i], pos2)
            mutated[i] = np.insert(temp, pos1 + 1, value)
    
    return mutated


def scramble_mutation(population, mutation_rate=0.01):
    """
    Perform scramble mutation on a permutation-based population.
    Randomly selects two points (p1, p2) in a chromosome and scrambles the genes in between.

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
            # pick two cut points
            p1, p2 = sorted(np.random.choice(chromosome_length, size=2, replace=False))
            subseq = mutated[i, p1+1:p2].copy()
            np.random.shuffle(subseq)
            mutated[i, p1+1:p2] = subseq

    return mutated


def inversion_mutation(population, mutation_rate=0.01):
    """
    Perform inversion mutation on a permutation-based population.
    Randomly selects two points (p1, p2) and reverses the genes in between.

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
            # pick two cut points
            p1, p2 = sorted(np.random.choice(chromosome_length, size=2, replace=False))
            mutated[i, p1+1:p2] = mutated[i, p1+1:p2][::-1]

    return mutated

#######################################
######-Replacement strategy-#######
#######################################

def plus_strategy(parents, parents_fitness, offspring, offspring_fitness, problem_type="min"):
    """
    (μ + λ) Replacement:
    Combine parents and offspring, then select the best μ individuals.
    Works for both minimization and maximization problems.
    """
    num = len(parents)
    combined_pop = np.vstack((parents, offspring))
    combined_fitness = np.concatenate((parents_fitness, offspring_fitness))

    if problem_type == "max":
        # Higher fitness = better
        top_indices = np.argsort(combined_fitness)[-num:]
    elif problem_type == "min":
        # Lower raw value = better → so take lowest
        top_indices = np.argsort(combined_fitness)[:num]
    else:
        raise ValueError("problem_type must be 'min' or 'max'")

    return combined_pop[top_indices]


def comma_strategy(offspring, offspring_fitness, num, problem_type="min"):
    """
    (μ, λ) Replacement:
    Only offspring are considered, select best μ.
    """
    if problem_type == "max":
        top_indices = np.argsort(offspring_fitness)[-num:]
    elif problem_type == "min":
        top_indices = np.argsort(offspring_fitness)[:num]
    else:
        raise ValueError("problem_type must be 'min' or 'max'")

    return offspring[top_indices]
