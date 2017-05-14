import Util
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
    sentiment_analysis.train(reviews[:-10000], labels[:-10000])

    ## Setting up the Recommender System Environment

    print("Reading dataset for Recommender System Setup")
    train_data, test_data = Util.read_data_from_csv(PATH, 'ua.base', 'ua.test')

    print("Creating recommender system")
    recommender = RecommenderSystem(train_data, test_data, PATH)

    print("Getting recommendation")
    movies_url = recommender.get_movies_url_for_user_prediction(user_id=1)

    ## Getting info from imdb api

    print("Getting ids")
    imdb_ids = Util.get_imdb_id_for(movies_url)

    print("Getting info from api")
    response = Util.get_title_plot(imdb_ids)

    print response

    ## sentiment analysis ##

    final_recomendation = []

    for movie in response.keys():
        sentiment = sentiment_analysis.run(response[movie])

        if sentiment == 'POSITIVE':
            final_recomendation.append(movie)

    print("Recommended movies based in similarity and plot sentiment: ")
    for m in final_recomendation:
        print(m)






