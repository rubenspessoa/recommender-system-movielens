import pandas as pd
import numpy as np

def read_data_from_csv(path, train_filename, test_filename):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv(path + train_filename, sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv(path + test_filename, sep='\t', names=r_cols, encoding='latin-1')

    return ratings_base, ratings_test

def read_all_data_from_csv(path):
    header = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    return pd.read_csv(path + 'u.data', sep='\t', names=header)

def create_user_item_matrices(train_data, test_data, path):
    df = read_all_data_from_csv(path)

    # Matrices sizing
    n_users = df.user_id.unique().shape[0]
    n_items = df.movie_id.unique().shape[0]

    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    return train_data_matrix, test_data_matrix