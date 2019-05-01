import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import scikitplot as skplt
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import pickle


### Reference https://stackabuse.com/text-classification-with-python-and-scikit-learn/

def process_txt_file(tweets_file):
    """
    Convert our txt tweet list to pandas dataframe
    :param tweets_file: raw txt tweet file
    :return: pandas dataframe
    """
    tweet_list = []
    tag_list = []
    if 'trump' in tweets_file.lower():
        tag = 0
    else:
        # 1 for AOC
        tag = 1
    with open(tweets_file) as o:
        tweets_json = json.load(o)
        for tweet in tweets_json:
            tweet_str = "".join(str(t) for t in tweet[3])
            tweet_list.append(tweet_str)
            tag_list.append(tag)

    tweet_df = pandas.DataFrame({'tweets': tweet_list, 'tag': tag_list})

    return tweet_df


def text_to_feature(count_vect, tfidf_transformer, x):
    x_counts = count_vect.fit_transform(x)
    x_tfidf = tfidf_transformer.fit_transform(x_counts)
    return x_tfidf


def training_process(all_df, count_vect, tfidf_transformer):
    """
    This function shuffles our training data,separates it to train and test, then trains the classifier.
    :param all_df:
    :param count_vect:
    :param tfidf_transformer:
    :return: trained model, test set
    """
    all_df = shuffle(all_df)
    x_train, x_test, y_train, y_test = train_test_split(all_df['tweets'], all_df['tag'], test_size=0.1)
    train_feature = text_to_feature(count_vect, tfidf_transformer, x_train)
    trained_model = MultinomialNB().fit(train_feature, y_train)
    return trained_model, x_test, y_test


def evaluation_score(evaluation_tweet_df, trained_eval_model, count_vect):
    predict = trained_eval_model.predict(count_vect.transform(evaluation_tweet_df["tweets"]))
    evaluation_tweet_df['predicted'] = predict

    score = cal_score(evaluation_tweet_df)
    return score


def evaluate_result_file(result_df, trained_eval_model, count_vect):
    result_df["tag"] = result_df.apply(mapping,axis=1)
    predict = trained_eval_model.predict(count_vect.transform(result_df["tweets"]))
    result_df['predicted'] = predict

    filenames = set(result_df['filename'])

    for filename in filenames:
        temp_df = result_df[result_df["filename"] == filename]
        score = cal_score(temp_df)
        # result_df[result_df["filename"] == filename].loc['score'] = score
        result_df.loc[result_df["filename"] == filename, "score"] = score

    return result_df

def mapping(row):
    if "trump" in row["filename"]:
        tag = 0
    else:
        tag = 1
    return tag

def cal_score(df):
    error = 0
    for tag, predict in zip(df['tag'], df['predicted']):
        if predict != tag:
            error += 1
    return (evaluation_tweet_df.shape[0] - error) / evaluation_tweet_df.shape[0]


def train(all_df, count_vect, tfidf_transformer):
    """
    This function takes care of all the training process, and dumps our trained file too.
    :param all_df: the aggregated dataset
    :param count_vect:
    :param tfidf_transformer:
    :return: trained model
    """
    trained_model, x_test, y_test = training_process(all_df, count_vect, tfidf_transformer)
    y_pred = trained_model.predict(count_vect.transform(x_test))
    print(accuracy_score(y_test, y_pred))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix of {}".format("NB"))
    pickle.dump(trained_model, open("binary_classifier/tweet_classifier.pkl", 'wb'))
    pickle.dump(count_vect, open("binary_classifier/count_vector.pkl", 'wb'))
    return trained_model
    # Uncomment here to see the confusion plot
    # plt.show()


if __name__ == '__main__':
    trump_df = process_txt_file("TRUMP_2600_tweets.txt")
    aoc_df = process_txt_file("ALL_AOC_TWEETS.txt")
    all_df = trump_df.append(aoc_df)
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    # You can either use the pre-trained model or train it yourself by just deleting the pre-trained pkl
    try:
        print("Using Loaded Model")
        trained_model = pickle.load(open('binary_classifier/tweet_classifier.pkl', 'rb'))
        count_vect = pickle.load(open("binary_classifier/count_vector.pkl", 'rb'))
    except FileNotFoundError:
        print("First Time Training")
        trained_model = train(all_df, count_vect, tfidf_transformer)

    # Here, we generate a score for all the files in the folder and write it to csv
    results = []
    filenames = []
    for filename in os.listdir("generated_text"):
        evaluation_tweet_df = pandas.read_csv(os.path.join("generated_text", filename))
        score = evaluation_score(evaluation_tweet_df, trained_model, count_vect)
        print("The evaluation score for  {} is {}".format(filename, str(score)))
        results.append(score)
        filenames.append(filename)

    results_df = pandas.DataFrame({'filename': filenames, 'result': results})
    results_df.to_csv("evaluation_result.csv")

    final_result_df = pandas.read_csv("result.csv",encoding = 'unicode_escape')
    evaluated_result_df = evaluate_result_file(final_result_df,trained_model,count_vect)
    # For some reason saving to csv here messed up with file.
    evaluated_result_df.to_excel("result_evaluated.xlsx", encoding='utf-8')
