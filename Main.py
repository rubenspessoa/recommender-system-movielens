import pandas as pd
import numpy as np
import Util

from RecommenderSystems import RecommenderSystems
from sklearn.metrics.pairwise import pairwise_distances

PATH = 'dataset-movielens/'


if __name__ == '__main__':

    train_data, test_data = Util.read_data_from_csv(PATH, 'ua.base', 'ua.test')

    recommender = RecommenderSystems()

    train_data_matrix, test_data_matrix = Util.create_user_item_matrices(train_data, test_data, PATH)
    print train_data_matrix

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

    item_prediction = recommender.predict(train_data_matrix, item_similarity, 'item')
    user_prediction = recommender.predict(train_data_matrix, user_similarity, 'user')

    user_id_2predict = 1
    user_recommendation = {}

    for enum, rate in enumerate(user_prediction[user_id_2predict]):
        if rate >= 1:
            movie_id = enum + 1
            user_recommendation[movie_id] = rate

    print user_recommendation

    i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
              'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(PATH + 'u.item', sep='|', names=i_cols,
                        encoding='latin-1')

    recommended_movies_url = []

    for line in items.itertuples():
        if line[0] in user_recommendation.keys():
            recommended_movies_url.append(line[5])

    print recommended_movies_url

    ## sentiment analysis ##





