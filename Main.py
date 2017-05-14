import Util, imdb
from RecommenderSystem import RecommenderSystem
from SentimentNetwork import SentimentNetwork
PATH = 'dataset-movielens/'


if __name__ == '__main__':

    ## Setting up the Sentiment Analysis Environment

    print("Reading dataset for Sentiment Analysis Setup")
    g = open('dataset-sa/reviews.txt', 'r')
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('dataset-sa/labels.txt', 'r')
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    print("Creating sentiment analysis")
    sentiment_analysis = SentimentNetwork(reviews[:-1000], labels[:-1000], polarity_cutoff=0.8, min_count=20,
                                          learning_rate=0.1)

    print("Training the neural net")
    sentiment_analysis.train(reviews[:-1000], labels[:-1000])

    ## Setting up the Recommender System Environment

    print("Reading dataset for Recommender System Setup")
    train_data, test_data = Util.read_data_from_csv(PATH, 'ua.base', 'ua.test')

    print("Creating recommender system")
    recommender = RecommenderSystem(train_data, test_data, PATH)

    print("Getting recommendation")
    movies_url = recommender.get_movies_url_for_user_prediction(user_id=1)

    ## Getting info from imdb api

    example = "http://www.imdb.com/title/tt0113189/"
    example = example.replace("http://www.imdb.com/title/tt", "")
    example = example.replace("/", "")

    imdb_access = imdb.IMDb()
    result = imdb_access.get_movie(example)

    title = result['canonical title']
    plot = result['plot']

    ## sentiment analysis ##

    sentiment_analysis.run(plot)





