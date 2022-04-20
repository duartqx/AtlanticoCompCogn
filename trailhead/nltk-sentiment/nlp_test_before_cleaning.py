from nltk import classify, FreqDist, NaiveBayesClassifier
from nltk.corpus import stopwords, twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from random import shuffle
from re import sub
from string import punctuation

# Since I've selected a custom folder to download all nltk_data, nltk's
# documentation mentions that it's required to export NLTK_DATA variable to the
# folder where the data is located. In my case that would be
# NLTK_DATA="./nltk_data"

#def lemmatize_sentences(tokens):
#    lemmatizer = WordNetLemmatizer()
#    lemmatized = []
#    for word, tag, in pos_tag(tokens):
#        if tag.startswith('NN'): pos = 'n'
#        elif tag.startswith('VB'): pos = 'v'
#        else: pos = 'a'
#        lemmatized.append(lemmatizer.lemmatize(word, pos))
#    return lemmatized

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        #token = sub("http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\'(?:%[0-9a-fA-F][0-9a-fA-F]))+","", token)
        token = sub("https?[:.]?\s?\/\/(?:\s*[^\/\s.]+)+(?:\s*\.\s*[^\/" + \
                "\s.]+)*(?:\s*\/\s*[^\/\s]+)*","", token)
        token = sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'): pos = 'n'
        elif tag.startswith('VB'): pos = 'v'
        else: pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in punctuation and \
                token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def clean_tokens(tokens, stop_words):
    return [remove_noise(t, stop_words) for t in tokens]

def get_all_words(cleaned_tokens):
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens):
    for tweet_tokens  in cleaned_tokens:
        yield dict([token, True] for token in tweet_tokens)

def create_dataset(sentiment, tokens_for_model):
    return [(tweet_dict, sentiment) for tweet_dict in tokens_for_model] 

if __name__ == '__main__':

    # Data tokenization
    #positive_tweets = twitter_samples.strings('positive_tweets.json')
    #negative_tweets = twitter_samples.strings('negative_tweets.json')
    #text = twitter_samples.strings('tweets.20150430-223406.json')
    #tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    #print(tweet_tokens)
    #Output:
    #['#FollowFriday', '@France_Inte', '@PKuchly57', '@Milipol_Paris', 'for',
    #'being', 'top', 'engaged', 'members', 'in', 'my', 'community', 'this',
    #'week', ':)']
    #print(tweet_tokens)[0] # Output: #FollowFriday
    
    #print(pos_tag(tweet_tokens))
    # Data normalization with lemmatize_sentence
    #print(lemmatize_sentences(tweet_tokens))

    # List of english's stopwords
    # Stopwords are the most common words in the language (like 'the', 'me')
    stop_words = stopwords.words('english')

    # Removing noise
    #print(remove_noise(tweet_tokens, stop_words))

    # Getting the tokens
    pst_tokens = twitter_samples.tokenized('positive_tweets.json')
    ngt_tokens = twitter_samples.tokenized('negative_tweets.json')

    # Generating word list with noise removed
    pst_clean_tokens = clean_tokens(pst_tokens, stop_words)
    ngt_clean_tokens = clean_tokens(ngt_tokens, stop_words)

    # Comparing results
    #print(pst_tokens[500]); print(pst_clean_tokens[500])

    # Determining word density and Frequency
    #all_pos_words = get_all_words(pst_clean_tokens)
    #print(*{word for word in all_pos_words}, sep='\n')
    #freq_dist_pos = FreqDist(all_pos_words)

    # Check for the most common  words
    #print(freq_dist_pos.most_common(10))
    # [(':)', 3691), (':-)', 701), (':d', 658), ('thanks', 388), ('follow',
    # 357), ('love', 333), ('...', 290), ('good', 283), ('get', 263), ('thank',
    # 253)]

    pst_tokens_for_model = get_tweets_for_model(pst_clean_tokens)
    ngt_tokens_for_model = get_tweets_for_model(ngt_clean_tokens)

    pst_dataset = create_dataset('Positive', pst_tokens_for_model)
    ngt_dataset = create_dataset('Negative', ngt_tokens_for_model)
    dataset = pst_dataset + ngt_dataset

    # Shuffles positives and negatives inside the dataset to avoid bias
    shuffle(dataset)

    # Slices train and test by 70:30
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)
    print('Accuracy is:', classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))
    # Output:
    # Accuracy is: 0.9946666666666667
    # Most Informative Features
    #                   :) = True           Positi : Negati =    993.8 : 1.0
    #                  bam = True           Positi : Negati =     22.4 : 1.0
    #             follower = True           Positi : Negati =     21.5 : 1.0
    #                  sad = True           Negati : Positi =     18.3 : 1.0
    #             followed = True           Negati : Positi =     16.5 : 1.0
    #              welcome = True           Positi : Negati =     15.9 : 1.0
    #                  via = True           Positi : Negati =     15.6 : 1.0
    #           appreciate = True           Positi : Negati =     15.1 : 1.0
    #                  x15 = True           Negati : Positi =     14.9 : 1.0
    #               arrive = True           Positi : Negati =     12.2 : 1.0
    
    custom_tweet = 'I ordered just once from TerribleCo, they screwed up, ' + \
            'never used the app again.'
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(classifier.classify(dict([token, True] for token in custom_tokens)))
    # Output: Negative

    custom_tweet = 'Congrats #SportStar on your 7th best goal from last ' + \
            'season winning goal of the year :) #Baller #Topbin ' + \
            '#oneofmanyworldies'
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(classifier.classify(dict([token, True] for token in custom_tokens)))
