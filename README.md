# twitter_bot_nlp_2019

For our final project for a Natural Language Processing class, we designed a Language Model using Keras that learns from a user's tweets and then generates similar langauge. We then use these tweets to create a bot that tweets out things similar to what the users would say. 

### Training Model + Generating Text 
Run the following comamnd: `models.py --type train  -tf PARSED_TWEETS_FILE_PATH -wf WEIGHTS_FILE_PATH`
There are a number of optional arguments that you can set: 
- -e: Number of epochs (default is 50)
- -do: Dropout (default is 0.3)
- -ea: Early stopping parameter, to have no early stopping use -1 (default is 0.1)
- -em: Embedding size (default is 100)
- -hs: Number of nodes in dense hidden layers (default is 100)

### Using Pre-Cached Weights 
Run the following comamnd: `models.py --type load  -tf PARSED_TWEETS_FILE_PATH -wf WEIGHTS_FILE_PATH`

### Scraping from Twitter + Using our Parser
There are two steps you'll have to do to get tweets into a suitable format for our model. 
1. Scrape tweets
Setup a config.py using the format from config_format.py. Then, run the following command: `python3 twitter_api.py --extract USER_NAME`

2. Parse tweets
Run the following command: `python3 parse_tweets.py --parse RAW_TWEET_FILE_PATH`
This sets up the tweets in a list of lists. Each tweet's list is of the format [TWEET_ID, TWEET_DATE, TWEET_TIME, TWEET_TEXT]. In our parsing, we made some decisions for how to tokenize tweets, for example:

- We added whitespace around most punctuation marks so that they are their own tokens. However, we left conjunctions in tact (it's, I'd.)
- We removed all links (which represent either quote tweets or images.) We replaced these with a MEDIA tag (defaults to "<MEDIA>".)
- We removed all "\n" characters within a tweet and instead added a newline indicator "<NL>" to preseve structure in the recreations. 
- We left in emojis. 
