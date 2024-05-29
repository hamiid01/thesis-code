import pandas as pd
import numpy as np


def concatenation(set1, set2, columns):
    columns_to_drop = ['class_label']
    set1_repeated = pd.concat([set1.drop(columns_to_drop, axis=1)] * len(set2), ignore_index=True)

    set2_repeated = set2.loc[set2.index.repeat(len(set1))].reset_index(drop=True)

    concatenated_set = pd.concat([set1_repeated, set2_repeated], axis=1)
    concatenated_set.columns = columns

    return concatenated_set
    
def main(df_train, columns, config):
"""
    -df_train: the original dataset
    -columns : the columns of the dataset to be generated
    -config : a dictionnary that contains :'pop_size', 'individual_size', 'num_generations', 'mutation_rate' and 'stagnation_limit'
    -population : the initial population, passed as an argument because it will be fixed for all versions of the GA
    -experminet : indicates which GA version to use
    this main function will return the concatenated dataset
    firstly we split the dataset to a set of majority class instances and a set of minority class instances,
    secondly we select Set_N randomly
    next we produce N_C and P_C
    in the end we concatenate those two and return the concatenated dataset."""


    N = df_train[df_train['class_label'] == df_train['class_label'].value_counts().idxmax()].reset_index(drop=True)
    P = df_train[df_train['class_label'] == df_train['class_label'].value_counts().idxmin()].reset_index(drop=True)


    Set_N = N.sample(config['individual_size'])

    N_C = concatenation(N, Set_N, columns)
    P_C = concatenation(P, P, columns)
    # GA is set True for P concatenation to avoid the problem of the weight column which
    # i s not present in P

    df_C = pd.concat([N_C, P_C]).reset_index(drop=True)

    return df_C
