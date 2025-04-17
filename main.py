from algorithms.genetic_algorithm import *
from timeit import default_timer as timer

start = timer()
pop_bin = initial_binary_population(10, 10)
print('Binary population:\n', pop_bin)
pop_real = initial_real_population(10, 10, -10, 10)
print('Real population:\n', pop_real)

fit_bin = fitness(pop_bin, "max")
fit_real = fitness(pop_real, "max")
print('Binary population fitness:', fit_bin)
print('Real population fitness:', fit_real)

end = timer()

print('Elapsed time', end - start) # time in seconds