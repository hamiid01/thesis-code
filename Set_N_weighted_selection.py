import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def concatenation1(set1, set2, columns):
    columns_to_drop = ['class_label', 'weight']
    set1_repeated = pd.concat([set1.drop(columns_to_drop, axis=1)] * len(set2), ignore_index=True)
    set2_repeated = set2.loc[set2.index.repeat(len(set1))].reset_index(drop=True).drop(['weight'], axis=1)

    concatenated_set = pd.concat([set1_repeated, set2_repeated], axis=1)
    concatenated_set.columns = columns

    return concatenated_set

def concatenation2(set1, set2, columns):
    columns_to_drop = ['class_label']
    set1_repeated = pd.concat([set1.drop(columns_to_drop, axis=1)] * len(set2), ignore_index=True)

    set2_repeated = set2.loc[set2.index.repeat(len(set1))].reset_index(drop=True)

    concatenated_set = pd.concat([set1_repeated, set2_repeated], axis=1)
    concatenated_set.columns = columns

    return concatenated_set
    
def weight_calculation(N, df_train):
    df_feature_train = df_train.drop('class_label', axis=1).values
    df_label_train = df_train['class_label'].values
    N_feature = N.drop('class_label', axis=1).values
    # Initialize NearestNeighbors with k=5
    nn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(df_feature_train)

    # Find the 5 nearest neighbors for instances in N within the total dataset
    distances, indices = nn.kneighbors(N_feature)

    # Apply the delta function: 1 if the neighbor is of the same class, 0 otherwise
    delta_function = lambda idx: (df_label_train[idx] == N['class_label'][0]).astype(
        int)  # returns 1 or 0 (True or False)

    weights = []
    for i in range(indices.shape[0]):
        # Calculate the weight for each instance in N
        weight = np.sum([delta_function(idx) for idx in indices[i, 1:]], axis=0) / 5.0
        # indices[i, 1:] is the the i-th row without the values of the 1st column
        # We skip the first neighbor because it's the point itself
        weights.append(weight)

    # Add weights to the DataFrame
    N['weight'] = weights
    
def Set_N_determination_baseline(df_train, N, set_N_size):
    weight_calculation(N, df_train)
    return N.sample(n=int(set_N_size), replace=False, weights='weight')
    
def main(df_train, columns, config):
    """this main function will return the concatenated dataset
    firstly we split the dataset to a set of majority class instances and a set of minority class instances,
    secondly we specify the cardinality of Set_N,
    thirdly depending on whether we are using the GA or not we determine the elements of Set_N
    next we produce N_C and P_C
    in the end we concatenate those two and return the concatenated dataset."""

    N = df_train[df_train['class_label'] == df_train['class_label'].value_counts().idxmax()].reset_index(drop=True)
    P = df_train[df_train['class_label'] == df_train['class_label'].value_counts().idxmin()].reset_index(drop=True)

    Set_N = Set_N_determination_baseline(df_train, N, config['individual_size'])

    N_C = concatenation1(N, Set_N, columns)
    P_C = concatenation2(P, P, columns)
    # GA is set True for P concatenation to avoid the problem of the weight column which
    # i s not present in P

    df_C = pd.concat([N_C, P_C]).reset_index(drop=True)

    return df_C

