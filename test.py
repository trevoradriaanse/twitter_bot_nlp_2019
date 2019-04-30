import json

with open("ALL_TRUMP_TWEETS.txt") as o:
    tweets_json = json.load(o)


tweets_new = []

i = 0
for tweet in tweets_json:
    if i< 500:
        tweets_new.append(tweet)
        i += 1
    else:
        break

with open('TRUMP_500_tweets.txt', 'w') as f:
    json.dump(tweets_new, f)
