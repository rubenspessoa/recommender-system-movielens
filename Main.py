import Util
from RecommenderSystem import RecommenderSystem
from SentimentNetwork import SentimentNetwork
PATH = 'dataset-movielens/'


if __name__ == '__main__':

    ## Setting up the Sentiment Analysis Environment

    print("1 - Reading dataset for Sentiment Analysis Setup")

    g = open('dataset-sa/reviews.txt', 'r')
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('dataset-sa/labels.txt', 'r')
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    print("2 - Creating Sentiment Analysis Class")
    sentiment_analysis = SentimentNetwork(reviews[:-1000], labels[:-1000], polarity_cutoff=0.8, min_count=20,
                                          learning_rate=0.1)

    print("3 - Training the neural net")
    sentiment_analysis.train(reviews[:-10000], labels[:-10000])

    print("4 - Reading dataset for Recommender System Setup")
    train_data, test_data = Util.read_data_from_csv(PATH, 'ua.base', 'ua.test')

    print("5 - Creating Recommender System Class")
    recommender = RecommenderSystem(train_data, test_data, PATH)

    print("6 - Getting recommendation for a given user_id")
    movies_url = recommender.get_movies_url_for_user_prediction(user_id=1)

    print("7 - Getting movie info from IMDb API")
    imdb_ids = Util.get_imdb_id_for(movies_url)
    response = Util.get_title_plot(imdb_ids)

    print("8 - Perform Sentiment Analysis on plots")
    final_recomendation = []

    for movie in response.keys():
        sentiment = sentiment_analysis.run(response[movie])

        if sentiment == 'POSITIVE':
            final_recomendation.append(movie)

    print("Recommended movies based in similarity and plot sentiment: ")
    for m in final_recomendation:
        print(m)






