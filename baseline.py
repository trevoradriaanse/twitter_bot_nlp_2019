from nltk import word_tokenize
import numpy as np
import codecs
from generate import GENERATE
from sklearn.preprocessing import normalize


# Adapted from Assignment 1


# Build a word set first
def return_set(tweets):
    result = []
    for tweet in tweets:
        tokens = word_tokenize(tweet)
        words = [word.lower() for word in tokens if word.isalpha()]
        result.extend(words)
    return list(set(result))


def make_dict(tweet_word):
    result = {}
    for counter, word in enumerate(tweet_word):
        result[word] = counter
    return result


def bigram(tweets, tweet_dict):
    counts = np.zeros(shape=(len(tweet_dict), len(tweet_dict)))
    counts += 0.1
    previous_word = '<s>'
    for tweet in tweets:
        tokens = word_tokenize(tweet)
        for token in tokens:
            if token.isalpha():
                counts[tweet_dict[previous_word.lower()]][tweet_dict[token.lower()]] += 1
                previous_word = token
    probs = normalize(counts, norm='l1', axis=1)
    probs = probs.transpose()
    # print(probs)
    return probs


if __name__ == '__main__':
    tweets = codecs.open("ALL_AOC_TWEETS.txt")
    tweet_word = return_set(tweets)
    tweet_dict = make_dict(tweet_word)
    probs = bigram(tweets, tweet_dict)
    # Generate
    for i in range(0, 9):
        print(GENERATE(tweet_dict, normalize(probs, norm='l1', axis=1), 'bigram', 8, "the") + "\n")
