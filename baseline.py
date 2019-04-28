import argparse
import sys
from nltk import word_tokenize
import numpy as np
import codecs
import json
from generate import GENERATE
from sklearn.preprocessing import normalize


# Adapted from Assignment 1


# Build a word set first
def return_set(tweets):
    result = []
    for tweet in tweets:
        words = tweet.split(" ")
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
        tokens = tweet.split(" ")
        for token in tokens:
            counts[tweet_dict[previous_word]][tweet_dict[token]] += 1
            previous_word = token
    probs = normalize(counts, norm='l1', axis=1)
    probs = probs.transpose()
    # print(probs)
    return probs

parser = argparse.ArgumentParser(description='Run the Baseline Model')
parser.add_argument('-tf','--tweets-filepath', default="", type=str, help='File path to load tweets')

def perplexity(tweets, tweet_dict, probs):
    perps = []
    tweet_prob = 1
    for tweet in tweets:
        tweet_len = len(tweet)
        for word in tweet.split():
            word_prob = probs[tweet_dict[word.lower()]]
            tweet_prob *= word_prob
        perplexity = 1/(pow(tweet_prob, 1.0/tweet_len))
        perps.append(perplexity)
    return perps

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.tweets_filepath) as o:
        tweets_json = json.load(o)
    tweets = [tweet[3] for tweet in tweets_json]
    tweet_word = return_set(tweets)
    tweet_dict = make_dict(tweet_word)
    probs = bigram(tweets, tweet_dict)
    perp = perplexity(tweets, tweet_dict, probs)

    # Generate
    for i in range(0, 9):
        print(GENERATE(tweet_dict, normalize(probs, norm='l1', axis=1), 'bigram', 8, "the") + "\n")
        print('Perplexity:', perp)
