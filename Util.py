from urllib2 import URLError

import pandas as pd

def read_data_from_csv(path, train_filename, test_filename):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv(path + train_filename, sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv(path + test_filename, sep='\t', names=r_cols, encoding='latin-1')

    return ratings_base, ratings_test

def read_all_data_from_csv(path):
    header = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    return pd.read_csv(path + 'u.data', sep='\t', names=header)

def get_imdb_id_for(movies_url):
    import urllib2

    imdb_ids = []

    for url in movies_url:
        try:
            response = urllib2.urlopen(unicode(url))
            new_url = response.geturl()
            new_url = new_url.replace("http://www.imdb.com/title/tt", "")
            new_url = new_url.replace("/", "")

            try:
                id = int(new_url)
                imdb_ids.append(new_url)
            except ValueError as e:
                pass

        except URLError as e:
            pass

    return imdb_ids

def get_title_plot(movie_ids):
    import imdb

    imdb_access = imdb.IMDb()
    response = {}

    for id in movie_ids:
        result = imdb_access.get_movie(id)
        title = result['canonical title']
        plot = result['plot'][0]
        response[title] = plot

    return response

