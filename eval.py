from baseline import return_set, make_dict, bigram
from models import prep_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np 
import json
import argparse
import sys
import ast
import csv
import difflib, distance


UNKNOWN_TOKEN = "<UNK>"

def unigram_probs(tweets):

    # create a list of all the indiviaul words in the tweets:
    words = [UNKNOWN_TOKEN]
    for tweet in tweets:
        tokens = tweet.split(" ")
        words.extend(tokens)
    vocab = list(set(words))

    # turns list of vocab into a dict that looks like {id: word}
    word_index_dict = {}
    for i, line in enumerate(vocab):
        word_index_dict[line.strip()] = i

    # get counts of each word
    counts = np.zeros(len(word_index_dict), dtype=int)
    for tweet in tweets:
        tokens = tweet.strip().split(" ")
        for token in tokens:
            counts[word_index_dict[token]] += 1
    counts[word_index_dict[UNKNOWN_TOKEN]] = 1

    probs = counts / np.sum(counts)
    return probs, word_index_dict


def perplexity(tweets, probs, word_index_dict):
    # calculate of all the tweets given the probabilites and word_index_dict 
    # these come from unigram_probs
    perps = []
    for tweet in tweets:
        tweet_prob = 1
        tokens = tweet.split(" ")
        tweet_len = len(tokens)
        for word in tokens:
            if word != "":
                if word_index_dict.get(word):
                    word_prob = probs[word_index_dict[word]]
                else: 
                    word_prob = probs[word_index_dict[UNKNOWN_TOKEN]]
                tweet_prob = tweet_prob * word_prob
        if tweet_prob != 0:
            perplexity = 1/(pow(tweet_prob, 1.0/tweet_len))
            perps.append(perplexity)
    return perps


def find_most_similar(corpus, tweet):
    # a function to gauge similarity. It uses python's difflib to calculate 
    # similarity between the given tweet (arg name: 'tweet') and each tweet 
    # in the corpus (arg name: 'corpus')
    # it also finds the tweet in the corpus most similar to the given tweet, 
    # and returns it as well as the match.
    similarity_max = 0
    best_tweet = ""
    best_match = [0,0,0]
    for corpus_tweet in corpus:
        seq = difflib.SequenceMatcher(None, corpus_tweet, tweet)
        sim = seq.ratio()
        if sim > similarity_max:
            similarity_max = sim
            best_tweet = corpus_tweet
        match = seq.find_longest_match(0, len(corpus_tweet), 0, len(tweet))
        if match[2] > best_match[2]:
            best_match = match
            best_matching_tweet = corpus_tweet

    j = best_match[1]
    k = best_match[2]
    best_match = tweet[j:j+k]
    return similarity_max, best_tweet, best_match, best_matching_tweet

def write_vals(csv_file, filename, perplexity_expected, perplexity_insample, perplexity_test, avg_sim, avg_len_longest_sim):
    fieldnames = ['filename', 'perplexity_expected', 'perplexity_insample', 'perplexity_test', 'avg_sim', 'avg_len_longest_sim']
    fieldnames = ['filename', 'avg_sim', 'avg_len_longest_sim']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerow({'filename': filename,
                         #'perplexity_expected': perplexity_expected,
                         #'perplexity_insample': perplexity_insample,
                         #'perplexity_test': perplexity_test,
                         'avg_sim': avg_sim,
                         'avg_len_longest_sim': avg_len_longest_sim
                        })


parser = argparse.ArgumentParser(description='Run the Keras Models')
parser.add_argument('-tf', '--tweet_filepath', type=str, help='Filepath to corpus of tweets')
parser.add_argument('-tt', '--test_tweets', type=str, help='Filepath to tweets to score')

if __name__ == '__main__':
    args = parser.parse_args()

    # Open the corpus, split into train and test.
    with open(args.tweet_filepath) as o:
        tweets_json = json.load(o)
    corpus_tweets = [tweet[3] for tweet in tweets_json]
    partition = int(.90*len(corpus_tweets))
    corpus_train = corpus_tweets[:partition]
    corpus_test = corpus_tweets[partition:]
    
    # Get unigram probabilities (to be used for perplexity calcs)
    probs, word_index_dict = unigram_probs(corpus_train)

    # Get our generated test tweets
    with open(args.test_tweets) as o:
        test_tweets = json.load(o)

    # Calculated "expected" perplexity on the held out corpus tweets
    # and calculate "actual" perplexity on our generated tweets.
    # (can be from baseline or a nn model)
    perplexity_expected = perplexity(corpus_test, probs, word_index_dict)
    perp_trained = perplexity(corpus_train, probs, word_index_dict)
    perplexity_test = perplexity(test_tweets, probs, word_index_dict)

    print("Trained perplexity has mean {} and variance {}".format(np.mean(perp_trained), np.std(perp_trained)))
    print("Expected perplexity has mean {} and variance {}".format(np.mean(perplexity_expected), np.std(perplexity_expected)))
    print("Actual perplexity has mean {} and variance {}".format(np.mean(perplexity_test), np.std(perplexity_test)))

    sim_vals = [1]
    longest_match_lens = [1]

    for test_tweet in test_tweets[0:10]:
        max_val, tweet, best_match, best_matching_tweet = find_most_similar(corpus_tweets, test_tweet)
        print("\n\n")
        print("Our tweet: {}".format(test_tweet))
        print("Most similar: {}".format(tweet))
        print("Tweet with longest match: {}".format(best_matching_tweet))
        print("Best match: {}".format(best_match))
        print("With similarity of: {}".format(max_val))
        sim_vals.append(max_val)
        longest_match_lens.append(len(best_match))

    with open('similarity_results.csv', 'a') as file:
        write_vals(file,
                   args.test_tweets,
                   np.mean(perplexity_expected),
                   np.mean(perp_trained),
                   np.mean(perplexity_test),
                   np.mean(sim_vals),
                   np.mean(longest_match_lens))
