from algorithms.genetic_algorithm import *
from timeit import default_timer as timer

start = timer()

import numpy as np

def genetic_algorithm(
    pop_size,
    chromosome_length,
    lower_bound,
    upper_bound,
    fitness_func,
    selection_method,
    crossover_func,
    mutation_func,
    replacement_func,
    mutation_rate=0.01,
    crossover_rate=0.75,
    a=0.5,
    max_fitness_calls=40000
):
    """
    Generic GA framework with stopping condition on total fitness evaluations.
    
    Arguments:
    pop_size -- population size (mu)
    chromosome_length -- number of genes in each individual
    lower_bound, upper_bound -- boundaries for real-valued chromosomes
    fitness_func -- function to minimize
    selection_method -- function to select parents
    crossover_func -- function to generate children
    mutation_func -- function to mutate children
    replacement_func -- function to generate next generation
    mutation_rate -- mutation probability
    crossover_rate -- crossover probability
    a -- crossover parameter
    max_fitness_calls -- stopping condition based on number of fitness evaluations
    
    Returns:
    best_solution -- best individual found
    best_fitness -- fitness of the best individual
    fitness_history -- list of best fitness per generation
    """
    
    # Initialize population
    population = initial_real_population(pop_size, chromosome_length, lower_bound, upper_bound)
    fitness_calls = 0
    fitness_history = []

    # Evaluate initial population
    raw, fit = fitness(population, fitness_func)
    fitness_calls += len(population)

    while fitness_calls < max_fitness_calls:
        # Select parents
        parents = selection_method(population, fit, num_parents=int(pop_size/2))

        # Generate offspring
        offspring = crossover_func(parents, a=a, crossover_rate=crossover_rate)
        offspring = mutation_func(offspring, mutation_rate=mutation_rate)

        # Evaluate offspring
        offspring_raw, offspring_fit = fitness(offspring, fitness_func)
        fitness_calls += len(offspring)

        # Replacement to form next generation
        if replacement_func.__name__ == "plus_strategy":
            population = replacement_func(population, fit, offspring, offspring_fit)
        elif replacement_func.__name__ == "comma_strategy":
            population = replacement_func(offspring, offspring_fit, pop_size)
        else:
            raise ValueError("Replacement function not recognized")

        # Record best fitness
        best_idx = np.argmax(fitness(population, fitness_func)[1])
        fitness_history.append(fitness(population, fitness_func)[1][best_idx])

    # Return best solution
    raw, fit = fitness(population, fitness_func)
    best_idx = np.argmax(fit)
    best_solution = population[best_idx]
    best_fitness = raw[best_idx]
    best_fit = fit[best_idx]
    print(best_fit)

    return best_solution, best_fitness, fitness_history

##############################

def adjiman(x):
    x1, x2 = x
    return np.cos(x1) * np.sin(x2) - x1 / (x2**2 + 1)

# --- GA Parameters ---
pop_size = 500
chromosome_length = 2       # x1 and x2
lower_bound = np.array([-1, -1])
upper_bound = np.array([2, 1])

# --- Run GA ---
best_solution, best_fitness, fitness_history = genetic_algorithm(
    pop_size=pop_size,
    chromosome_length=chromosome_length,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    fitness_func=adjiman,
    selection_method=proportional_selection,  # or tournament_selection, random_selection, etc.
    crossover_func=whole_arithmetic_crossover,  # or simple_crossover, whole_arithmetic_crossover
    mutation_func=complement_mutation,
    replacement_func=plus_strategy,  # or comma_strategy
)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

end = timer()
print("---")
print('Elapsed time:', end - start, 'seconds')
