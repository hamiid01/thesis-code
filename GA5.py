import numpy as np
import logging
import plot_GA_evaluation

import selection_function_roulette_wheel
import crossover_1_point
import fitness_function1

import mutation_function
import select_potential_indecies
import initial_population
import time


def genetic_algorithm(N, P, config, population):
    best_individual = None
    best_fitness = float('inf')
    #    previous_best_fitness = float('inf')
    #    stagnation_counter = 0
    fitness_history = []
    fitness_history1 = []
    global_population = [population]

    #    global_population.append(population)

    for generation in range(config['num_generations']):
        start_time = time.time()
        if len(population) != config['pop_size']:
            logging.error("population size %s", len(population))
        for ind in population:
            if len(ind) != config['individual_size']:
                logging.error("ind size %s", len(ind))
        
        # Fitness Evaluation
        fitness = [fitness_function1.synthesize_dataset(individual, N, P) for individual in population]
        fitness_history.append(min(fitness))
        fitness_history1.append((min(fitness), np.mean(fitness), max(fitness)))
        # list of fitness values of the individuals in the population
        # [a, b, c, ..., z]

        #  Elitism
        # sort the population according to the fitness
        sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
        elites = [ind for ind, _ in sorted_population]

        # Selection
        parents = selection_function_roulette_wheel.roulette_wheel_selection(population, fitness, int(len(population)//1.5))
        # parents = [ [, , , , ], ..., [, , , , ] ]

        # crossover
        offspring = crossover_1_point.crossover(parents)

        # generation of the new popultion
        population = offspring + elites[:config['pop_size'] - len(offspring)]
        
        #mutation
        mutation_function.mutate_fixed_cardinality(population, N.shape[0])
        global_population.append(population)

        #  Update the best solution found
        current_best_fitness, current_best_individual = sorted_population[0][1], sorted_population[0][0]
        if current_best_fitness < best_fitness:
            logging.info("New best fitness found: %f", current_best_fitness)
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        end_time = time.time()
        logging.info("Generation %d complete. Best fitness: %f", generation, best_fitness)
        print(f'time taken in generation {generation} : {end_time - start_time:.4f} seconds')

    Set_N = N.loc[best_individual]

    print(f'best_individual, {best_individual}')
    print(f'Set_N {N.loc[best_individual]}')

    plot_GA_evaluation.plot_fitness(fitness_history)
    plot_GA_evaluation.plot_fitness_details(fitness_history1)
    #    #plot_population_heatmap(global_population, len(potential_indices))
    #    #plot_GA_evaluation.plot_gene_variability_over_generations(global_population, len(potential_indices))

    return Set_N


def main(df_train, N, P, config, population):

            """
    df_train : the original dataset
    N: majority class samples
    P : majority class samples
    population: initial population
    #[ [a, b, c, e, f], [., ., ., ., .], [ ], ...., [ ], [x, y, z, t, u] ] : population
    #[a, b, c, e, f] : indivudual
    """

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Initializing GA with population size: %s", config['pop_size'])


    Set_N = genetic_algorithm(N, P, config, population)

    return Set_N
