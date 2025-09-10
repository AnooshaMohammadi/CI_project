import numpy as np
import pandas as pd
import benchmarkfcns as bf
from algorithms.genetic_algorithm import *
from algorithms.benchmark import benchmark_functions
from timeit import default_timer as timer

# =============================
# GA Function
# =============================
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
        best_solution, best_fitness, _ = genetic_algorithm(
            pop_size=50,
            chromosome_length=chromosome_length,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fitness_func=f.name,  # pass the actual function object
            selection_method=truncation_selection,
            crossover_func=simple_crossover,
            mutation_func=complement_mutation,
            replacement_func=plus_strategy,
            problem_type="min",
            mutation_rate=0.1,
            crossover_rate=0.75,
            a=0.5,
            max_fitness_calls=400
        )
        results.append(best_fitness)

    return np.mean(results), np.std(results)


# =============================
# Table generators
# =============================
def generate_info_table(func_list, filename):
    data = []
    for func in func_list:
        data.append({
            "Name": func.name,
            "Dimension": func.dimension,
            "Global Min": func.global_minima
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df


def generate_results_table(func_list, filename):
    data = []
    for idx, f in enumerate(func_list, start=1):
        print(f.name)  # shows the function being processed
        avg, std = run_ga_on_function(f)
        data.append({
            "No": idx,
            "Function": f.name,       # changed from f["name"] to f.name
            "GA Average": avg,
            "GA Std": std
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df



# =============================
# Run everything
# =============================
start = timer()

unimodal_funcs = [f for f in benchmark_functions if f.type == "unimodal"]
multimodal_funcs = [f for f in benchmark_functions if f.type == "multimodal"]

print("Generating unimodal info table...")
generate_info_table(unimodal_funcs, "unimodal_info.csv")

print("Generating multimodal info table...")
generate_info_table(multimodal_funcs, "multimodal_info.csv")

print("Running GA on unimodal functions...")
generate_results_table(unimodal_funcs, "unimodal_results.csv")

print("Running GA on multimodal functions...")
generate_results_table(multimodal_funcs, "multimodal_results.csv")

end = timer()
print(f"All done! Elapsed time: {end - start:.2f} seconds")
