import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def weight_calculation(df_train, N):
    # Extract features and labels from the training dataframe
    df_feature_train = df_train.drop('class_label', axis=1).values
    df_label_train = df_train['class_label'].values
    N_feature = N.drop('class_label', axis=1).values

    # Initialize Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(df_feature_train)

    # Find the 5 nearest neighbors for instances in N within the total dataset
    distances, indices = nn.kneighbors(N_feature)

    # Calculate weights
    weights = []
    delta_function = lambda idx: (df_label_train[idx] == N['class_label'][0]).astype(int)    #returns 1 or 0 (True or False)
    for i in range(indices.shape[0]):
        # Calculate the weight for each instance in N
        weight = np.sum([delta_function(idx) for idx in indices[i, 1:]], axis=0) / 5.0
        # indices[i, 1:] is the the i-th row without the values of the 1st column
        weights.append(weight)


    # Select indices of N with highest weights
    high_weight_indices = np.argsort(weights)[::-1][:len(weights) // 2]  # Select top 50% weighted instances

    return list(high_weight_indices)
