import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
#def plot_population_heatmap(population,minlength, title="Population Heatmap Over Generations"):
    # Assuming population is a list of lists, where each sublist represents a generation.
    #population_matrix = np.array([np.bincount(individual, minlength=minlength) for individual in population])  
    # Adjust minlength to the size of your gene pool
    #plt.figure(figsize=(12, 8))
    #ax = sns.heatmap(population_matrix, annot=False, cmap="YlGnBu")
    #ax.set_title(title)
    #ax.set_xlabel("Gene Index")
    #ax.set_ylabel("Generation")
    #plt.show()
"""
# TODO: fix this population heatmap function

def plot_fitness_details(fitness_history):
    # fitness_history is a list of lists with (min, avg, max) fitness per generation
    generations = list(range(1, len(fitness_history) + 1))
    fitness_df = pd.DataFrame(fitness_history, columns=['Min Fitness', 'Avg Fitness', 'Max Fitness'], index=generations)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=fitness_df)
    plt.title('Fitness Details Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(title='Fitness Metrics')
    plt.grid(True)
    plt.show()


def plot_gene_variability_over_generations(population, total_genes):
    # total_genes is the total number of possible genes in the genetic code

    # Prepare data for DataFrame: Collect data for each gene across all generations
    data = {f'Gene_{i}': [] for i in range(total_genes)}
    for generation in population:
        for individual in generation:
            # Extend individual gene data with None for missing genes if necessary
            extended_genes = individual + [None] * (total_genes - len(individual))
            for i in range(total_genes):
                data[f'Gene_{i}'].append(extended_genes[i])

    # Create DataFrame for plotting
    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(20, 10))  # Adjust the size as necessary
    sns.boxplot(data=df)
    plt.xticks(rotation=90, ha="right")
    plt.title("Variability of Gene Values Over Generations")
    plt.ylabel("Gene Value")
    plt.xlabel("Gene Index")
    plt.grid(True)
    plt.show()





def plot_population_heatmap(generations, minlength, title="Population Heatmap Over Generations"):
    # Assuming generations is a list of lists of lists (three levels: generations, individuals, genes).
    # visualize each generation's population by combining all individuals' genes in that generation.

    # Flatten the generations into a matrix where each row is a generation
    # and columns represent the count of each gene index.
    population_matrix = np.array([np.bincount([gene for individual in generation
                                               for gene in individual], minlength=minlength)
                                  for generation in generations])

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(population_matrix, annot=False, cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xlabel("Gene Index")
    ax.set_ylabel("Generation")
    plt.show()


def plot_fitness(fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, marker='o')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()
