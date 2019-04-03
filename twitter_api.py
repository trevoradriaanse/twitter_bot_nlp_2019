'''

Twitter API Interaction
@Author Trevor Adriaanse

'''

import tweepy as tw
import config

auth = tw.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# send a tweet
#api.update_status('Test')

def send_tweet(tweet):
    api.update_status(tweet)

def scrape_tweets(user):
    pass
