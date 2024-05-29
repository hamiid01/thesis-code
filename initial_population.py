from random import shuffle, sample


def initialize_population(pop_size, potential_indices, individual_size):
    """ Initialize a diverse population by creating unique random subsets of indices from the majority class. """
    population = []
    for _ in range(pop_size):
        shuffle(potential_indices)
        population.append(sample(potential_indices, individual_size))
    return population
