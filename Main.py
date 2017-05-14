import pandas as pd
import Util
from RecommenderSystem import RecommenderSystem
import imdb

PATH = 'dataset-movielens/'


if __name__ == '__main__':

    train_data, test_data = Util.read_data_from_csv(PATH, 'ua.base', 'ua.test')

    recommender = RecommenderSystem(train_data, test_data, PATH)

    movies_url = recommender.get_movies_url_for_user_prediction(user_id=1)

    print movies_url


    ## access link in internet
    ## parse url to get id
    example = "http://www.imdb.com/title/tt0113189/"

    example = example.replace("http://www.imdb.com/title/tt", "")
    example = example.replace("/", "")

    print example

    imdb_access = imdb.IMDb()
    result = imdb_access.get_movie(example)
    print result['canonical title']
    print result['plot']

    #plot outline
    #plot








    ## sentiment analysis ##





