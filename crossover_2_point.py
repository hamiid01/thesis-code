import pandas as pd
import numpy as np
from random import randint, sample

 
   
def crossover(parents):
    """ Perform two-point crossover on the selected parents to generate offspring. """
    offspring = []
    while len(offspring) < len(parents):
        p1, p2 = sample(parents, 2)  # Pick two random parents

        points = sorted(sample(range(1, len(p1)), 2)) #randomly select the 2 points of crossover 
        
        child1 = p1[:points[0]] + [x for x in p2[points[0]:points[1]] if x not in p1 ] + p1[points[1]:]
        child2 = p2[:points[0]] + [x for x in p1[points[0]:points[1]] if x not in p2 ] + p2[points[1]:]

        # Make sure offspring are of the right size by adding missing indices
        while len(child1) < len(p1):
            needed = len(p1) - len(child1)
            to_add = sample([x for x in p1 if x not in child1], needed)
            child1.extend(to_add)

        while len(child2) < len(p2):
            needed = len(p2) - len(child2)
            to_add = sample([x for x in p1 if x not in child2], needed)
            child2.extend(to_add)
            
        offspring.append(child1)
        offspring.append(child2)

    return offspring