import pandas as pd
import numpy as np
from random import sample


def tournament_selection(population, fitnesses,parents_size, tournament_size=3):
    """ Select parents using tournament selection, which helps maintain diversity and avoids premature convergence. """
    
    selected_parents = []
    for _ in range(parents_size):
        participants = sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(participants, key=lambda x: x[1])[0]
        selected_parents.append(winner)
    
    return selected_parents
