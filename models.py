import argparse
import sys
import random

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku 
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
        #input_sequences.append(token_list)
        # adding n-grams
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,   
                          maxlen=max_sequence_len, padding='pre'))


    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words, tokenizer


def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, args.embedding_size, input_length=input_len))
    model.add(LSTM(args.hidden_size))
    model.add(Dropout(args.dropout))
    model.add(Dense(total_words, activation='softmax'))
    return model


def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_sequence_len-1, padding='pre')
        probs = model.predict(token_list, verbose=0)
        next_tokens = sorted([i[1] for i,v in np.ndenumerate(probs)], key=lambda i:np.asarray(probs[0])[i], reverse=True)[:10]
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


parser = argparse.ArgumentParser(description='Run the Keras Models')
# one of the following is required:
parser.add_argument('-t', '--train', type=str, help='Train a new model')
parser.add_argument('-l', '--load', type=str, help='Load a cached model')
# args for the model
parser.add_argument('-tf','--tweets-filepath', default="", type=str, help='File path to load tweets')
parser.add_argument('-wf','--weights-filepath', default="new-file.txt", type=str, help='Load or store weights')

parser.add_argument('-e','--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
parser.add_argument('-ea','--early-stopping', default=0.1, type=float, help='Early stopping criteria')
parser.add_argument('-em','--embedding-size', default=100, type=int, help='Embedding dimension size')
parser.add_argument('-hs','--hidden-size', default=100, type=int, help='Hidden layer size')


#parser.set_defaults(retweet=False)

if __name__=='__main__':
    if len(sys.argv) < 1:
        args.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if not (args.train or args.load):
        parser.error('Set at least one of --train or --load')

    with open(args.tweets_filepath) as o:
        tweets_json = json.load(o)
    tweets = [tweet[3] for tweet in tweets_json]
    X, Y, max_len, total_words, tokenizer = prep_data(tweets)
    model = create_model(X, Y, max_len, total_words)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if args.train:
        # Train a new model
        checkpoint = ModelCheckpoint(args.weights_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='loss', min_delta=args.early_stopping)
        callbacks_list = [checkpoint, early_stopping]
        model.fit(X, Y, epochs=args.epochs, verbose=1, callbacks=callbacks_list)
        
    if args.load:
        # Load a model from file 
        model.load_weights(args.weights_fielpaths)

    for _ in range(10):
        print(generate_text("<s>", 50, max_len, model, tokenizer))
