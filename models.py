# coding:utf-8
import argparse
import sys
import random
import csv
import os
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.utils import multi_gpu_model
import keras.utils as ku
from tensorflow.contrib.keras.api.keras.initializers import Constant
import numpy as np
import json


# ref: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

def prep_data(tweets):
    '''
    Data is a list of tweets. 
    '''
    tokenizer = Tokenizer(num_words=None, filters='"%*+-=[]^_`{|}~\t\n', lower=False, split=' ')
    tokenizer.fit_on_texts(tweets)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for tweet in tweets:
        token_list = tokenizer.texts_to_sequences([tweet])[0]
        # input_sequences.append(token_list)
        # adding n-grams
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    # label = np.zeros((label, total_words), dtype=np.int8)
    return predictors, label, max_sequence_len, total_words, tokenizer


def create_model(predictors, label, max_sequence_len, total_words, regular):
    input_len = max_sequence_len - 1
    model = Sequential()

    if regular == "regular":
        print("Training with LSTM")
        model.add(Embedding(total_words, args.embedding_size, input_length=input_len))
    else:
        print("Training with Glove")
        word_to_index, index_to_word, word_to_embedding = read_glove_file('glove.twitter.27B.25d.txt')
        pretrained_embedding = create_pretrained_embedding_layer(word_to_embedding, word_to_index, False)
        model.add(pretrained_embedding)
    model.add(LSTM(args.hidden_size))
    model.add(Dropout(args.dropout))
    model.add(Dense(total_words, activation='softmax'))
    return model


# transfer learning with GloVe
# cf.: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
# https://stackoverflow.com/questions/48677077/how-do-i-create-a-keras-embedding-layer-from-a-pre-trained-word-embedding-datase

def read_glove_file(glove_file):
    with open(glove_file, encoding='utf-8') as f:
        word_to_embedding = {}
        word_to_index = {}
        index_to_word = {}

        for line in f:
            record = line.strip().split()
            token = record[0]
            word_to_embedding[token] = np.array(record[1:], dtype=np.float64)

        tokens = sorted(word_to_embedding.keys())
        for idx, tok in enumerate(tokens):
            keras_index = idx + 1  # 0 is reserved for masking in Keras
            word_to_index[tok] = keras_index
            index_to_word[keras_index] = tok

    return word_to_index, index_to_word, word_to_embedding


def create_pretrained_embedding_layer(word_to_embedding, word_to_index, is_trainable):
    vocab_len = len(word_to_index) + 1  # adding 1 to account for masking
    embedding_dim = next(iter(word_to_embedding.values())).shape[0]

    embedding_matrix = np.zeros((vocab_len, embedding_dim))
    for word, index in word_to_index.items():
        try:
            embedding_matrix[index, :] = word_to_embedding[word]
        except ValueError:
            pass
    embedding_layer = Embedding(vocab_len,
                                embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                trainable=is_trainable)
    return embedding_layer


def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=
        max_sequence_len - 1, padding='pre')
        probs = model.predict(token_list, verbose=0)
        next_tokens = sorted([i[1] for i, v in np.ndenumerate(probs)], key=lambda i: np.asarray(probs[0])[i],
                             reverse=True)[:10]
        output_word = ""
        predicted = next_tokens[random.randint(0, 2)]
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

        if output_word == "</s>":
            return seed_text
    return seed_text


def load_model_weights(model):
    model.load_weights("weights.best.hdf5")


def write_to_csv(args, result_text, result_file, loss):
    """
    This function writes the result to csv for analysis
    :param env:
    :param iter:
    :param reward:
    :return:
    """

    if os.path.exists(result_file):
        mode = False
        with open(result_file, mode='a') as csv_file:
            write_helper(args, csv_file, result_text, mode, loss)
    else:
        mode = True
        with open(result_file, mode='w+') as csv_file:
            write_helper(args, csv_file, result_text, mode, loss)


def write_helper(args, csv_file, result_text, mode, loss):
    fieldnames = ['epochs', 'dropout', 'model', 'earlystopping', 'dropout', 'embedding', 'hidden', 'filename',
                  'result_text', 'loss']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if mode:
        writer.writeheader()
    writer.writerow({'epochs': args.epochs, 'dropout': args.dropout, 'model': args.glove
                        , 'earlystopping': args.early_stopping
                        , 'embedding': args.embedding_size, 'hidden': args.hidden_size,
                     'filename': args.weights_filepath
                        , 'result_text': result_text, 'loss': str(loss)})


parser = argparse.ArgumentParser(description='Run the Keras Models')
# one of the following is required:
parser.add_argument('-t', '--type', type=str, help='Train a new model')
# args for the model
parser.add_argument('-g', '--glove', default="regular", type=str, help='Train with LSTM or Glove')
parser.add_argument('-tf', '--tweets-filepath', default="", type=str, help='File path to load tweets')
parser.add_argument('-wf', '--weights-filepath', default="new-file.txt", type=str, help='Load or store weights')
parser.add_argument('-e', '--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('-do', '--dropout', default=0.3, type=float, help='Dropout rate')
parser.add_argument('-ea', '--early-stopping', default=0.1, type=float, help='Early stopping criteria')
parser.add_argument('-em', '--embedding-size', default=100, type=int, help='Embedding dimension size')
parser.add_argument('-hs', '--hidden-size', default=100, type=int, help='Hidden layer size')

# parser.set_defaults(retweet=False)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        args.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if not (args.type):
        parser.error('Set --type to "train" or "load"')

    with open(args.tweets_filepath) as o:
        tweets_json = json.load(o)
    tweets = [tweet[3] for tweet in tweets_json]
    X, Y, max_len, total_words, tokenizer = prep_data(tweets)
    model = create_model(X, Y, max_len, total_words, args.glove)

    try:
        model = multi_gpu_model(model)
    except:
        pass

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    loss = []
    if args.type == "train":
        # Train a new model
        checkpoint = ModelCheckpoint(args.weights_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='loss', min_delta=args.early_stopping)
        callbacks_list = [checkpoint, early_stopping]
        history = model.fit(X, Y, epochs=args.epochs, verbose=1, callbacks=callbacks_list)
        loss = history.history['loss']
        # model.save(args.weights_filepath)
    if args.type == "load":
        # Load a model from file 
        model.load_weights(args.weights_filepath)

    for _ in range(10):
        generated_text = generate_text("<s>", 50, max_len, model, tokenizer)
        print(generated_text)
        write_to_csv(args, generated_text, 'result.csv', np.min(loss))
