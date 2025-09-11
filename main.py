import numpy as np
import pandas as pd
import benchmarkfcns as bf
from algorithms.genetic_algorithm import *
from algorithms.pso_algorithm import *
from algorithms.benchmark import benchmark_functions
from timeit import default_timer as timer

# =============================
# GA Function
# =============================
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
    max_fitness_calls=400
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


# =============================
# Benchmark functions metadata
# =============================

# =============================
# GA runner
# =============================
def run_ga_on_function(f, num_runs=20):
    lower_bound = np.array([f.range[0]] * f.dimension)
    upper_bound = np.array([f.range[1]] * f.dimension)
    chromosome_length = f.dimension

    results = []
    for _ in range(num_runs):
        best_solution, best_fitness, _ = genetic(
            pop_size=50,
            chromosome_length=chromosome_length,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fitness_func=f.name,  # pass the actual function object
            selection_method=proportional_selection,
            crossover_func=simple_arithmetic_crossover,
            mutation_func=complement_mutation,
            replacement_func=plus_strategy,
            problem_type="min",
            mutation_rate=0.01,
            crossover_rate=0.75,
            a=0.5,
            max_fitness_calls=40000
        )
        results.append(best_fitness)
    
    return np.mean(results), np.std(results), np.min(results)


def run_pso_on_function(f, num_runs=20):
    """
    Run PSO on a given function multiple times and return statistics.
    
    Arguments:
    f -- function object representing the benchmark function
    num_runs -- number of times to run PSO
    
    Returns:
    avg_fitness -- average best fitness across all runs
    std_fitness -- standard deviation of best fitness across all runs
    best_fitness -- best best fitness across all runs
    """
    lower_bound = np.array([f.range[0]] * f.dimension)
    upper_bound = np.array([f.range[1]] * f.dimension)
    dim = f.dimension

    results = []
    for _ in range(num_runs):
        best_position, best_fitness, _ = pso(
            pop_size=50,
            dim=dim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            velocity_lower_bound=-1,
            velocity_upper_bound=1,
            fitness_func=f.name,
            problem_type="min",
            max_fitness_calls=40000
        )
        results.append(best_fitness)
    
    avg_fitness = np.mean(results)
    std_fitness = np.std(results)
    best_fitness = np.min(results)
    
    return avg_fitness, std_fitness, best_fitness


# =============================
# Table generators
# =============================
def generate_info_table(func_list, filename):
    data = []
    for idx, func in enumerate(func_list, start=1):
        data.append({
            "Fn": f"F{idx}",
            "Name": func.name,
            "Range": func.range,
            "Dimension": func.dimension,
            "Global Min": func.global_minima
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df


def generate_results_table(func_list, filename):
    # Phase 1: Collect all results
    ga_avgs = []
    ga_stds = []
    pso_avgs = []
    pso_stds = []

    # First pass: collect data without ranks
    results = []

    for idx, f in enumerate(func_list, start=1):
        fn_name = f.name
        print(f"Processing {fn_name}")

        # Run GA
        print("Running GA...")
        ga_avg, ga_std, _ = run_ga_on_function(f)
        ga_avgs.append(ga_avg)
        ga_stds.append(ga_std)

        # Run PSO
        print("Running PSO...")
        pso_avg, pso_std, _ = run_pso_on_function(f)
        pso_avgs.append(pso_avg)
        pso_stds.append(pso_std)
        print("Done!")
        # Store raw data
        results.append({
            "Fn": f"F{idx}",
            "Stats": "Avg",
            "GA": ga_avg,
            "PSO": pso_avg
        })
        results.append({
            "Fn": f" ",
            "Stats": "Std",
            "GA": ga_std,
            "PSO": pso_std
        })
        results.append({
            "Fn": f" ",
            "Stats": "Rank",
            "GA": None,  # Placeholder for GA rank
            "PSO": None  # Placeholder for PSO rank
        })

    # Phase 2: Compute ranks based on proximity to global_minima
    ga_ranks = []
    pso_ranks = []

    for idx, f in enumerate(func_list):
        # Convert global_minima to float if necessary
        try:
            global_minima = float(f.global_minima)
        except ValueError:
            # Handle scientific notation or other formats
            global_minima = float(eval(f.global_minima))
        
        ga_diff = abs(ga_avgs[idx] - global_minima)
        pso_diff = abs(pso_avgs[idx] - global_minima)

        # Combine differences and sort
        combined_diffs = [(ga_diff, "GA"), (pso_diff, "PSO")]
        sorted_diffs = sorted(combined_diffs, key=lambda x: x[0])

        # Assign ranks based on sorted differences
        ga_rank = sorted_diffs.index((ga_diff, "GA")) + 1
        pso_rank = sorted_diffs.index((pso_diff, "PSO")) + 1

        ga_ranks.append(ga_rank)
        pso_ranks.append(pso_rank)

    # Fill in ranks
    rank_idx = 2  # Start at the third entry (index 2) for each function
    for idx in range(len(func_list)):
        results[rank_idx]["GA"] = ga_ranks[idx]
        results[rank_idx]["PSO"] = pso_ranks[idx]
        rank_idx += 3  # Move to the next function's rank entry

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return df


# =============================
# Run everything
# =============================
start = timer()

unimodal_funcs = [f for f in benchmark_functions if f.type == "unimodal"]
multimodal_funcs = [f for f in benchmark_functions if f.type == "multimodal"]
#multi = multimodal_funcs[-2:]

print("Generating unimodal info table...")
generate_info_table(unimodal_funcs, "unimodal_info.csv")

print("Generating multimodal info table...")
generate_info_table(multimodal_funcs, "multimodal_info.csv")

print("Running algorithms on unimodal functions...")
generate_results_table(unimodal_funcs, "unimodal_results.csv")

print("Running algorithms on multimodal functions...")
generate_results_table(multimodal_funcs, "multimodal_results.csv")

end = timer()
print(f"All done! Elapsed time: {end - start:.2f} seconds")