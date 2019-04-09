def parse_old_corpus_tweets(filename='RAW_CORPUS.txt'):
    '''
    Parses tweets specifically from "RAW_CORPUS.txt", a corpus of Donald Trump's tweets 
    between __ and ___. 
    
    Textfile format is tab separated with headings:  
    STATUSID    DATE    TIME    SOURCE  Retweet Favourite   TWEETS
    '''
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        header = f.readline()
        line = f.readline()
        while line:
            if line != "\n":
                line = line.strip()
                tweet_id, date, time, _, _, _, text = line.split("\t")
                tweet_id = tweet_id[1:].strip()
                text = text.replace('\xa0', ' ')
                tweets.append([tweet_id, date, time, text])

            line = f.readline()
    return tweets

def parse_twitter_api_tweets(filename):
    '''
    Parses tweets in the format that we use in our twitter scraping
    (see twitter_api.py -> scrape_tweets())
    '''
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        header = f.readline()
        line = f.readline()
        while line:
            if line != "\n":
                line = line.strip()
                tweet_id, date, time, text = line.split("\t")
                text = text.replace('\xa0', ' ')
                tweets.append([tweet_id, date, time, text])

            line = f.readline()
    return tweets

def get_donald_trump_tweets():
    all_tweets = []
    old_tweets = parse_old_corpus_tweets()
    new_tweets = parse_twitter_api_tweets('realdonaldtrump_raw_tweets.txt')
    # todo: get unique 
    #for tweet_list in old_tweets: 
    #    all_tweets.append(tweet_list[3])
    for tweet_list in new_tweets: 
        all_tweets.append(tweet_list[3])
    all_tweets = clean_tweets(all_tweets)
    return all_tweets

def clean_tweets(tweets):
    '''
    -> Replaces any link (signifies quote tweet or photo) with <MEDIA/> tag.
    -> Adds start and end tags for tweets. 
    '''
    import re
    cleaned_tweets = []
    for tweet in tweets:
        tweet = re.sub('https://[^\s]+', '<MEDIA/>', tweet)
        tweet = "<s> " + tweet + " </s>"
        cleaned_tweets.append(tweet)

    return cleaned_tweets

