import numpy as np

def initialize_pbest(positions, fitness_values):
    """Initialize personal best positions and fitness."""
    pbest_positions = positions.copy()
    pbest_fitness = fitness_values.copy()
    return pbest_positions, pbest_fitness

def initialize_gbest(pbest_positions, pbest_fitness, problem_type="min"):
    """Initialize global best from personal bests."""
    if problem_type == "min":
        best_idx = np.argmin(pbest_fitness)
    else:
        best_idx = np.argmax(pbest_fitness)
    return pbest_positions[best_idx], pbest_fitness[best_idx]

def update_velocity(velocities, positions, pbest_positions, gbest_position, w, c1, c2):
    """Update velocity of particles."""
    r1, r2 = np.random.rand(*positions.shape), np.random.rand(*positions.shape)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)
    return w * velocities + cognitive + social

def update_position(positions, velocities, lower_bound, upper_bound):
    """Update positions and keep them within bounds."""
    positions += velocities
    return np.clip(positions, lower_bound, upper_bound)

def update_pbest(positions, fitness_values, pbest_positions, pbest_fitness, problem_type="min"):
    """Update personal best if current fitness is better."""
    for i in range(len(positions)):
        if (problem_type == "min" and fitness_values[i] < pbest_fitness[i]) or \
           (problem_type == "max" and fitness_values[i] > pbest_fitness[i]):
            pbest_positions[i] = positions[i].copy()
            pbest_fitness[i] = fitness_values[i]
    return pbest_positions, pbest_fitness

def update_gbest(pbest_positions, pbest_fitness, gbest_position, gbest_fitness, problem_type="min"):
    """Update global best based on personal bests."""
    if problem_type == "min":
        best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[best_idx] < gbest_fitness:
            return pbest_positions[best_idx], pbest_fitness[best_idx]
    else:
        best_idx = np.argmax(pbest_fitness)
        if pbest_fitness[best_idx] > gbest_fitness:
            return pbest_positions[best_idx], pbest_fitness[best_idx]
    return gbest_position, gbest_fitness
