from random import choices

def roulette_wheel_selection(population, fitnesses, num_parents):
    #[ [a, b, c, e, f], [., ., ., ., .], [ ], ...., [ ], [x, y, z, t, u] ] : population
    #[a, b, c, ..., z] : fitnesses
    total_fitness = sum(fitnesses)
    selection_probs = [1-(fitness / total_fitness) for fitness in fitnesses]
    selected_indices = choices(range(len(population)), weights=selection_probs, k=int(num_parents))
    
    return [population[index] for index in selected_indices]
    # returns : [ [, , , , ], ..., [, , , , ] ] : length = num_parents