import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
import time
import logging


# Function to get columns of the concatenated dataset
def columns_of_concatenated_ds(df):
    c = list(df.columns.drop(['class_label']))
    columns = c + [e+'_N' for e in c] + ['class_label']
    return columns

# Function to concatenate two sets
def concatenation(set1, set2, columns):
    set1_repeated = pd.concat([set1.drop(['class_label'], axis=1)] * len(set2), ignore_index=True)
    set2_repeated = set2.loc[set2.index.repeat(len(set1))].reset_index(drop=True)
    concatenated_set = pd.concat([set1_repeated, set2_repeated], axis=1)
    concatenated_set.columns = columns
    return concatenated_set

#Function to producet the concatenated dataset
def synthesize_dataset(Set_N_indices, N, P):
    Set_N = N.iloc[Set_N_indices]
    columns = columns_of_concatenated_ds(N)
    N_C = concatenation(N, Set_N, columns)
    P_C = concatenation(P, P, columns)

    train_C = pd.concat([N_C, P_C]).reset_index(drop=True)
    train_C_features = train_C.drop(['class_label'], axis=1).values
    N_C_features = N_C.drop(['class_label'], axis=1).values
    P_C_features = P_C.drop(['class_label'], axis=1).values
    y = train_C['class_label'].values

    negative = np.bincount(y).argmax()
    positive = np.bincount(y).argmin()

    return calculate_simplified_complexity(train_C_features, y, N_C_features, P_C_features, negative, positive)

def calculate_simplified_complexity(train_C_features, y, N_C_features, P_C_features, negative, positive, k=5):
    #start_time = time.time()
    nn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(train_C_features)
    distances, indices = nn.kneighbors(P_C_features)
    distances2, indices2 = nn.kneighbors(N_C_features)
    
    neighbor_classes = y[indices[:, 1:]]
    majority_count = np.sum(neighbor_classes == negative, axis=1)
    neighbor_classes2 = y[indices2[:, 1:]]
    minority_count = np.sum(neighbor_classes2 == positive, axis=1)

    average_majority_proportion = np.mean(majority_count / k)
    average_minority_proportion = np.mean(minority_count / k)
    #end_time = time.time()
    #logging.info(f"neighborhood measure done : {end_time - start_time:.4f} seconds")
    
    complexity_score = 0.5 * average_majority_proportion + 0.5 * average_minority_proportion

    return complexity_score



