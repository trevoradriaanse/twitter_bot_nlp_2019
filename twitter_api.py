'''

Twitter API Interaction
@Author Trevor Adriaanse

'''

import argparse
import tweepy as tw
import config
import sys

auth = tw.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

parser = argparse.ArgumentParser(description='Interact with Twitter API')
parser.add_argument('-s', '--send', type=str, help='Send given tweet')
parser.add_argument('-x', '--extract', type=str, help='Extract tweets of given user')
grp = parser.add_mutually_exclusive_group(required=False)
grp.add_argument('-r', '--retweets', dest='retweet', action='store_true', help='Retain retweets')
grp.add_argument('-nr', '--no_retweets', dest='retweet', action='store_false', help='Omit retweets')
parser.set_defaults(retweet=False)

def send_tweet(tweet):
    api.update_status(tweet)

def scrape_tweets(user):
    if user.startswith('@'):
        user = user[1:]
    with open(user+'_raw_tweets.txt', 'w') as f:
        for tweet in tw.Cursor(api.user_timeline, tweet_mode='extended', screen_name=user).items():
            date, time = tuple(str(tweet.created_at).split())
            if not args.retweet and tweet.full_text.startswith('RT'):
                pass
            else:
                f.write(tweet.id_str+'\t'+date+'\t'+time+'\t'+tweet.full_text+'\n')          

if __name__=='__main__':
    if len(sys.argv) < 2:
        args.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if not (args.send or args.extract):
        parser.error('Set at least one of --send or --extract')
    if args.send:
        send_tweet(args.send)
    if args.extract:
        scrape_tweets(args.extract)

