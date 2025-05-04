from algorithms.genetic_algorithm import *
from timeit import default_timer as timer

start = timer()
pop_bin = initial_binary_population(3, 3)
pop_real = initial_real_population(3, 3, -10, 10)
#print('Binary population:\n', pop_bin)
print('Real population:\n', pop_real)

fit_bin = fitness(pop_bin, "min")
fit_real = fitness(pop_real, "min")

print("---")
#print('Binary population fitness:', fit_bin)
print('Real population fitness:', fit_real)

selected_rand = random_selection(pop_bin, 2)
selected_pro_bin = proportional_selection(pop_bin, fit_bin, 2)
selected_rank = rank_based_selection(pop_bin, fit_bin, 2, "min")
selected_tru = truncation_selection(pop_bin, fit_bin, 2, 90, "min")
#print("random selected parents (binary):\n", selected_rand)
#print("prportionaly selected parents (bin):\n", selected_pro_bin)
#print("rank-based selected parents (bin):\n", selected_rank)
#print("truncation selected parents (bin):\n", selected_tru)

selected_tru_real = truncation_selection(pop_real, fit_real, 2, 90, "min")
print("truncation selected parents (real):\n", selected_tru_real)

print("---")
print("---")

cross_simple_child = simple_crossover(selected_tru_real, 0.3)

print("children produced using simple cross over:\n", cross_simple_child)
print("---")

end = timer()

print('Elapsed time:', end - start) # time in seconds