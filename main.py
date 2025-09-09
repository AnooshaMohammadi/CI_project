from algorithms.genetic_algorithm import *
from timeit import default_timer as timer

start = timer()

# --- Initialize populations ---
pop_bin = initial_binary_population(50, 30)
pop_real = initial_real_population(50, 30, -10, 10)

print('Binary population:\n', pop_bin)
print('Real population:\n', pop_real)

# --- Evaluate fitness ---
# fitness() now returns raw_values, fitness_scores
raw_bin, fit_bin = fitness(pop_bin, lambda x: np.sum(x**2))  # example sum-of-squares for binary
raw_real, fit_real = fitness(pop_real, lambda x: np.sum(x**2))  # example sum-of-squares for real

print("---")
print('Binary population fitness (raw values):', raw_bin)
print('Binary population fitness (selection scores):', fit_bin)
print('Real population fitness (raw values):', raw_real)
print('Real population fitness (selection scores):', fit_real)

# --- Selection ---
selected_rand = random_selection(pop_bin, 2)
selected_pro_bin = proportional_selection(pop_bin, fit_bin, 2)
selected_rank = rank_based_selection(pop_bin, fit_bin, 2)
selected_tru = truncation_selection(pop_bin, fit_bin, 2, 90)

print("---")
print("Random selected parents (binary):\n", selected_rand)
print("Proportional selected parents (binary):\n", selected_pro_bin)
print("Rank-based selected parents (binary):\n", selected_rank)
print("Truncation selected parents (binary):\n", selected_tru)

selected_tru_real = truncation_selection(pop_real, fit_real, 30, 90)
print("Truncation selected parents (real):\n", selected_tru_real)

# --- Crossover ---
cross_simple_child = simple_crossover(selected_tru_real)
print("---")
print("Children produced using simple crossover:\n", cross_simple_child)

end = timer()
print("---")
print('Elapsed time:', end - start, 'seconds')
