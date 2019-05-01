import argparse
import sys
from nltk import word_tokenize
import numpy as np
import codecs
import json
from generate import GENERATE
from sklearn.preprocessing import normalize
import csv


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
    """
    This function calculates bigram. This is based on homework 1
    :param tweets:
    :param tweet_dict:
    :return:
    """
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


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.tweets_filepath) as o:
        tweets_json = json.load(o)
    tweets = [tweet[3] for tweet in tweets_json]
    tweet_word = return_set(tweets)
    tweet_dict = make_dict(tweet_word)
    probs = bigram(tweets, tweet_dict)

    # Generate
    tweets = []
    tags = []
    baseline_filenames = []
    if 'trump' in args.tweets_filepath.lower():
        tag = 0
        baseline_filename = "trump_baseline"+args.tweets_filepath
    else:
        # 1 for AOC
        tag = 1
        baseline_filename = "aoc_baseline"+args.tweets_filepath

    # Here we write our baseline model generated text to our result file.
    with open('result.csv', mode="a", encoding='utf-8') as result_file:
        writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, 9):
            tweet = GENERATE(tweet_dict, normalize(probs, norm='l1', axis=1), 'bigram', 8, "the")
            print(tweet)
            writer.writerow(['NA','NA','baseline_bigram','NA','NA','NA','NA',baseline_filename,str(tweet),'NA'])
        result_file.close()

