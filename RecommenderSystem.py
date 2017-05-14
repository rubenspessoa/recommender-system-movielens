# coding: utf-8

import numpy as np
import pandas as pd
import Util

from sklearn.metrics.pairwise import pairwise_distances

class RecommenderSystem:

    train_data_matrix = []
    test_data_matrix = []
    user_similarity_matrix = []
    item_similarity_matrix = []
    item_prediction = []
    user_prediction = []
    PATH = ''

    def __init__(self, train_data, test_data, path):
        self.train_data_matrix, self.test_data_matrix = self.create_user_item_matrices(train_data, test_data, path)
        self.calculate_similarities()
        self.create_prediction_matrices()
        self.PATH = path

    def predict(self, ratings, similarity, type):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
                [np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred

    def calculate_similarities(self):
        self.user_similarity_matrix = pairwise_distances(self.train_data_matrix, metric='cosine')
        self.item_similarity_matrix = pairwise_distances(self.train_data_matrix.T, metric='cosine')

    def create_user_item_matrices(self, train_data, test_data, path):
        df = Util.read_all_data_from_csv(path)

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

    def create_prediction_matrices(self):
        self.item_prediction = self.predict(self.train_data_matrix, self.item_similarity_matrix, 'item')
        self.user_prediction = self.predict(self.train_data_matrix, self.user_similarity_matrix, 'user')

    def get_prediction_for(self, user_id):
        user_recommendation = {}

        for enum, rate in enumerate(self.user_prediction[user_id]):
            if rate >= 1:
                movie_id = enum + 1
                user_recommendation[movie_id] = rate

        return user_recommendation

    def get_movies_url_for_user_prediction(self, user_id):
        prediction = self.get_prediction_for(user_id)

        i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
                  'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items = pd.read_csv(self.PATH + 'u.item', sep='|', names=i_cols,
                            encoding='latin-1')

        recommended_movies_url = []

        for line in items.itertuples():
            if line[0] in prediction.keys():
                recommended_movies_url.append(line[5])

        return recommended_movies_url