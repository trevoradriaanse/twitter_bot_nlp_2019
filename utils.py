import json
import re

START = "<s> "
END = " </s>"
MEDIA = "<MEDIA>"


def parse_twitter_api_tweets(filename, tweets=[], ids=[]):
    '''
    Parse output from scrape_tweets in twitter_api.py
    Adds some tweet formatting 
    '''
    with open(filename, 'r', encoding='UTF-8') as f:
        header = f.readline()
        line = f.readline()

        while line:
            if line != "\n":
                line = line.strip()
                # Depending on the corpus format, split differently 
                if len(line.split("\t")) == 7:
                    tweet_id, date, time, _, _, _, text = line.split("\t")
                    tweet_id = tweet_id[1:].strip()
                else:
                    tweet_id, date, time, text = line.split("\t")

                if tweet_id not in ids: # filter out overlapping tweets
                    ids.append(tweet_id)
                    # replace a few special chars that show up weird in the file:
                    text = text.replace('\xa0', ' ')
                    text = text.replace('&amp;', '&')
                    text = text.replace('\u2019', r"'")
                    text = text.replace('\u201c', r'"')
                    text = text.replace('\u201d', r'"')
                    # Replace any URL with a tag:
                    text = re.sub('https://[^\s]+', MEDIA, text)
                    # add whitespace before & after punctuation:
                    text = re.sub("[^A-Za-z0-9\s#@<>']", ' \g<0> ', text);
                    text = START + text
                    tweet = text + END
                    tweets.append([tweet_id, date, time, tweet])

                    # next line
                    line = f.readline()

                else: # We have already seen this tweet! 
                    line = f.readline()

    return tweets, ids

tweets, ids = parse_twitter_api_tweets("AOC_raw_tweets.txt")

with open('ALL_AOC_TWEETS.txt', 'w', encoding='utf-8') as outfile:  
    json.dump(tweets, outfile)
