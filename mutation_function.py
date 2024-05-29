from random import random, randint, sample


def mutate_fixed_cardinality(population, N_size, mutation_rate=0.05):
    for individual in population:
        if random() < mutation_rate:
            index_to_replace = randint(0, len(individual) - 1)
            new_index = sample([i for i in range(N_size) if i not in individual], 1)[0]
            individual[index_to_replace] = new_index
    return population


