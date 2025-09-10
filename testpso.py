from algorithms.genetic_algorithm import initial_real_population, fitness
from algorithms.pso_algorithm import *
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt


start = timer()

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
    max_fitness_calls=400
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
    print("positions:",positions)
    print("velocities",velocities)

    fitness_values = fitness(positions, fitness_func)
    
    pbest_positions, pbest_fitness = initialize_pbest(positions, fitness_values)
    gbest_position, gbest_fitness = initialize_gbest(pbest_positions, pbest_fitness, problem_type)
    
    fitness_history = []

    for _ in range(max_fitness_calls):
        # Update velocities and positions
        velocities = update_velocity(velocities, positions, pbest_positions, gbest_position, w, c1, c2)
        positions = update_position(positions, velocities, lower_bound, upper_bound)

        # Evaluate new fitness
        fitness_values = fitness(positions, fitness_func)

        # Update pbest and gbest
        pbest_positions, pbest_fitness = update_pbest(positions, fitness_values, pbest_positions, pbest_fitness, problem_type)
        gbest_position, gbest_fitness = update_gbest(pbest_positions, pbest_fitness, gbest_position, gbest_fitness, problem_type)

        fitness_history.append(gbest_fitness)

    return gbest_position, gbest_fitness, fitness_history

##########test#####################

def sphere(x):
    """Simple benchmark: minimize sum of squares"""
    return np.sum(x**2)

if __name__ == "__main__":
    start = timer()

    # Parameters
    num_particles = 30
    dim = 5
    lower_bound = -5.12
    upper_bound = 5.12

    # Run PSO
    gbest_position, gbest_fitness, fitness_history = pso(
        pop_size=num_particles,
        dim=dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        velocity_lower_bound=-1,
        velocity_upper_bound=1,
        fitness_func=sphere,   # Change to rastrigin for more challenge
        problem_type="min"
    )

    end = timer()

    # Print results
    print("Best Position:", gbest_position)
    print("Best Fitness:", gbest_fitness)
    print("Execution Time:", end - start, "seconds")

    # Plot convergence
    plt.plot(fitness_history, label="Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("PSO Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()