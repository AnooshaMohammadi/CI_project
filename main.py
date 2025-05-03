from algorithms.genetic_algorithm import *
from timeit import default_timer as timer

start = timer()
pop_bin = initial_binary_population(3, 3)
print('Binary population:\n', pop_bin)
pop_real = initial_real_population(3, 3, -10, 10)
print('Real population:\n', pop_real)

fit_bin = fitness(pop_bin, "min")
fit_real = fitness(pop_real, "min")

print("---")
print('Binary population fitness:', fit_bin)
print('Real population fitness:', fit_real)

selected_rand = random_selection(pop_bin, 2)
selected_pro_real = proportional_selection(pop_real, fit_real, 2)
print("random selected parents (binary):\n", selected_rand)
print("prportionaly selected parents (real):\n", selected_pro_real)
print("---")
end = timer()

print('Elapsed time:', end - start) # time in seconds