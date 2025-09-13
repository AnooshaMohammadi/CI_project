from algorithms.utility import *
from algorithms.genetic_algorithm import *
from algorithms.pso_algorithm import *
from timeit import default_timer as timer
import numpy as np
#import matplotlib.pyplot as plt


def pso(
    pop_size,
    dim,
    lower_bound,
    upper_bound,
    velocity_lower_bound,
    velocity_upper_bound,
    fitness_func,
    w=0.74,
    c1=1.42,
    c2=1.42,
    problem_type="min",
    max_fitness_calls=40000
):
    """
    Generic PSO framework with stopping condition on total fitness evaluations.
    
    Arguments:
    pop_size -- number of particles
    dim -- dimensionality of search space
    lower_bound, upper_bound -- search space boundaries
    fitness_func -- objective function to minimize/maximize
    w -- inertia weight
    c1, c2 -- cognitive and social coefficients
    problem_type -- "max" for maximization, "min" for minimization
    max_fitness_calls -- stopping condition based on number of fitness evaluations
    
    Returns:
    best_solution -- best particle found
    best_fitness -- fitness of the best particle
    fitness_history -- list of best fitness per iteration
    """
    # --- Initialization ---
    positions = initial_real_population(pop_size, dim, lower_bound, upper_bound)
    velocities = initial_real_population(pop_size, dim, velocity_lower_bound, velocity_upper_bound)
    #print("positions:",positions)
    #print("velocities",velocities)

    fitness_values = fitness(positions, fitness_func)
    
    pbest_positions, pbest_fitness = initialize_pbest(positions, fitness_values)
    gbest_position, gbest_fitness = initialize_gbest(pbest_positions, pbest_fitness, problem_type)
    
    fitness_history = []
    fitness_calls = 0
    fitness_calls += len(positions)

    while fitness_calls < max_fitness_calls:
        # Update velocities and positions
        velocities = update_velocity(velocities, positions, pbest_positions, gbest_position, w, c1, c2)
        positions = update_position(positions, velocities, lower_bound, upper_bound)

        # Evaluate new fitness
        fitness_values = fitness(positions, fitness_func)
        fitness_calls += len(positions)
        # Update pbest and gbest
        pbest_positions, pbest_fitness = update_pbest(positions, fitness_values, pbest_positions, pbest_fitness, problem_type)
        gbest_position, gbest_fitness = update_gbest(pbest_positions, pbest_fitness, gbest_position, gbest_fitness, problem_type)

        fitness_history.append(gbest_fitness)

    return gbest_position, gbest_fitness, fitness_history


def genetic(
    pop_size,
    chromosome_length,
    lower_bound,
    upper_bound,
    fitness_func,
    selection_method,
    crossover_func,
    mutation_func,
    replacement_func,
    problem_type="min",
    mutation_rate=0.1,
    crossover_rate=0.75,
    a=0.5,
    max_fitness_calls=40000
):
    # Initialize population
    population = initial_real_population(pop_size, chromosome_length, lower_bound, upper_bound)
    fitness_calls = 0
    fitness_history = []

    # Evaluate initial population
    fit = fitness(population, fitness_func)
    fitness_calls += len(population)

    while fitness_calls < max_fitness_calls:
        # Selection
        parents = selection_method(population, fit, num_parents=int(pop_size / 2))

        # Variation
        offspring = crossover_func(parents, a=a, crossover_rate=crossover_rate)
        offspring = mutation_func(offspring, mutation_rate=mutation_rate)

        # Evaluate offspring
        offspring_fit = fitness(offspring, fitness_func)
        fitness_calls += len(offspring)

        # Replacement
        if replacement_func.__name__ == "plus_strategy":
            population = replacement_func(population, fit, offspring, offspring_fit)
        elif replacement_func.__name__ == "comma_strategy":
            population = replacement_func(offspring, offspring_fit, pop_size)
        else:
            raise ValueError("Replacement function not recognized")

        # Record best fitness
        if problem_type == "max":
            best_idx = np.argmax(fit)
        else:  # min
            best_idx = np.argmin(fit)
        fitness_history.append(fit[best_idx])

        # Update fitness
        fit = fitness(population, fitness_func)

    # Final best solution
    if problem_type == "max":
        best_idx = np.argmax(fit)
    else:
        best_idx = np.argmin(fit)

    best_solution = population[best_idx]
    best_fitness = fit[best_idx]
    return best_solution, best_fitness, fitness_history



best_position_pso, best_fitness_pso, fitness_history_pso = pso(
    pop_size = 50,
    dim = 2,
    lower_bound = -5.12,
    upper_bound = 5.12,
    velocity_lower_bound = -1,
    velocity_upper_bound= 1,
    fitness_func = "sphere",
    w=0.74,
    c1=1.42,
    c2=1.42,
    problem_type="min",
    max_fitness_calls=40000
)

best_position_ga, best_fitness_ga, fitness_history_ga = genetic(
    pop_size = 50,
    chromosome_length = 2,
    lower_bound = -5.12,
    upper_bound = 5.12,
    fitness_func = "sphere",
    selection_method = proportional_selection,
    crossover_func = simple_arithmetic_crossover,
    mutation_func = swap_mutation,
    replacement_func = plus_strategy,
    problem_type="min",
    mutation_rate=0.1,
    crossover_rate=0.75,
    a=0.5,
    max_fitness_calls=40000
)


start = timer()
print("ga best position:", best_position_ga)
print("ga best fitness:", best_fitness_ga)
print("--------------------")
print("pso best position:", best_position_pso)
print("pso best fitness:", best_fitness_pso)

end = timer()
print(f"All done! Elapsed time: {end - start:.2f} seconds")