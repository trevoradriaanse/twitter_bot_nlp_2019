from baseline import return_set, make_dict, bigram
from models import prep_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
import json
import argparse
import sys
import ast

### PERPLEXITY ###
def perplexity(tweets, tweet_dict, probs):
    perps = []
    tweet_prob = 1
    for tweet in tweets:
        tweet_len = len(tweet)
        for word in tweet.split():
            word_prob = probs[tweet_dict[word]]
            tweet_prob *= word_prob
        perplexity = 1/(pow(tweet_prob, 1.0/tweet_len))
        perps.append(perplexity)
    return perps
### END PERPLEXITY ###

### COSINE SIMILARITY ###
def create_vectors(tweets):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(tweets)
    return tfidf

def cosine_similarity(v1, v2):
    sim = cosine_similarity_sklearn(v1, v2)
    return sim

### END COSINE SIMILARITY  ###



### TWEET CLASSIFIER ###


def create_test_and_train(data, target='trump'):
    '''
    Input: a path to a text file containing a list of lists
    Output: files for training and testing, using a 90/10 split
    '''
    contents = ''
    with open(data) as f:
        trump_contents = f.read()
    tmp1 = ast.literal_eval(trump_contents)
    data1 = np.array(tmp1)
    labels = np.chararray(len(tmp1))
    labels[:] = target[0]
    X1, y1 = data1, trump_labels
    # split 90/10
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=42)
    np.save(f'{target}_train_feat', X1_train)
    np.save(f'{target}_test_feat', X1_test)
    np.save(f'{target}_train_labels', y1_train)
    np.save(f'{target}_test_labels', y1_test)
    

def create_classifier(vocab, labels):
    classifier = Sequential([
        # input_dim: "Size of the vocabulary", output_dim: "Dimension of the dense embedding"
        Embedding(input_dim=len(vocab)+1, output_dim=args.embedding_size, input_length=None),
        # units: "dimensionality of the output space"
        Bidirectional(LSTM(units=args.hidden_size, return_sequences=False)),
        # units: "dimensionality of the output space"
        Dense(units=len(labels), activation='softmax'),
    ])
    return classifier


#### Next functions from A4 NLP taught by Nathan Schneider ###
def get_vocabulary_and_data(data_file, split, max_vocab_size=None):
    vocab = Counter()
    data = []
    labels = []
    with open(data_file, 'r', encoding='utf8') as f:
        for line in f:
            cols = line.split(',')
            s, surname, label = cols[0].strip(), cols[1].strip(), cols[2].strip()
            if s==split:
                surname = list(surname)
                surname = [START]+surname+[END]
                data.append(transform_text_sequence(surname))
                labels.append(label)
            for tok in surname:
                vocab[tok]+=1

    vocab = sorted(vocab.keys(), key=lambda k: vocab[k], reverse=True)
    if max_vocab_size:
        vocab = vocab[:len(vocab)-max_vocab_size-4]
    vocab = [UNK, PAD] + vocab

    return {k:v for v,k in enumerate(vocab)}, set(labels), data, labels

def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]

def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]

def one_hot_encode_label(label):
    vec = [1.0 if l==label else 0.0 for l in labels]
    return np.array(vec)

def batch_generator(data, labels, vocab, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for doc, label in zip(data,labels):
            batch_x.append(vectorize_sequence(doc, vocab))
            batch_y.append(one_hot_encode_label(label))
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []
### TWEET CLASSIFIER ###

parser = argparse.ArgumentParser(description='Run the Keras Models')
parser.add_argument('-t', '--type', type=str, help='Evaluate through comparison or individually')

# If Comparison (Trump v AOC):
parser.add_argument('-tr', '--trump_fp', type=str, help='Filepath to Trump Corpus')
parser.add_argument('-ao', '--aoc_fp', type=str, help='Filepath to AOC Corpus')
parser.add_argument('-te', '--test', type=str, help='Filepath to tweets to classify')
parser.add_argument('-em', '--embedding-size', type=int, default=100, help='Embedding Size')
parser.add_argument('-hs', '--hidden-size', type=int, default=100, help='Hidden Layer Size')


# If Individual Evaluation (Perplexity, Similarity):
parser.add_argument('-tf', '--tweet_filepath', type=str, help='Filepath to corpus of tweets')
parser.add_argument('-tt', '--test_tweets', type=str, help='Filepath to tweets to score')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.type == 'individual':
        with open(args.tweet_filepath) as o:
            tweets_json = json.load(o)
        corpus_tweets = [tweet[3] for tweet in tweets_json]
        tweet_word = return_set(corpus_tweets)
        tweet_dict = make_dict(tweet_word)
        probs = bigram(corpus_tweets, tweet_dict)

        with open(args.test_tweets) as o:
            test_tweets = json.load(o)

        perplexity_expected = perplexity(corpus_tweets, tweet_dict, probs)
        perpelxity_test = perplexity(test_tweets, tweet_dict, probs)
        print("Expected: ", (perplexity_expected), "Actual: ", (perpelxity_test))


        sim = cosine_similarity(create_vectors(corpus_tweets), create_vectors(test_tweets))
        print(sim)

    if args.type == 'compare':

        create_test_and_train(args.trump_fp, args.aoc_fp)
        model = create_classifier(vocab, 2)
        classifer = KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)
        # evaluate with cross-validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=12345)
        results = cross_val_score(estimator, X, dummy_y, cv=kfold)
        print("Accurary is: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        # build evaluator to differentiate between AOC and Trump 
        # how well does it classify it's own tweets? 
        # how well does it classify ours? 

