from random import sample


def gene_pool_crossover(parents):
    # parents = [[, , , , ], ..., [, , , , ]] 
    offsprings = []
    while len(offsprings) < len(parents):
        p1, p2 = sample(parents, 2)  # Pick two random parents
        gene_pool = list(set(p1 + p2))  #create the genetic pool 
        
        offspring1 = sample(gene_pool, len(p1))
        offspring2 = sample(gene_pool, len(p1))
        
        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return offsprings
# The function returns a list of offspring individuals, each constructed by randomly sampling
# the gene pool without replacement, ensuring each gene in an offspring is unique.
