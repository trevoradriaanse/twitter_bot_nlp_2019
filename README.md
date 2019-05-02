# twitter_bot_nlp_2019

For our final project for a Natural Language Processing class, we designed a Language Model using Keras that learns from a President Trump's tweets and then generates similar langauge. We then use these tweets to create a bot that tweets out things similar to what the users would say. 


## File Inventory
- **binary_classifier**:
  - This folder contains pkl file for trained classfier and fitted count vector
- **generated_text**:
  - This folder contains csv files generated by our trained model using models.py's loading mode. 
  - Some of the weights couldn't be correctly loaded. https://github.com/keras-team/keras/issues/10417 We think this might be a keras issue. Thus, their generated csv file is not included. For the evaluation of those models, we can reference to our similarity/perplexity and binary-classification score of 10 tweets generated while training. 
- **trained_model**:
  - This folder only contains trained lstm models. All the models' weights can be found in shared google drive https://drive.google.com/file/d/1e75TyZUZRCJBOITx04_yB-8Lf9KLLGG2/view?usp=sharing
- **baseline.py**:
  - This py file computes bigram and generate text. Then it writes the results to result.csv. 
- **binary_classifier.py**:
  - This py file builds a binary text classifier using actual President Trump's and Congresswoman Alexandria Ocasio-Cortez's originial tweets.
  - This file will then go through all the file in generated_text folder and calculate an accuracy score for each of them. This results is compiled in to evaluation_result.csv
  - This file will also go through all the tweets in our result.csv file and calculate a score for each of the model. The new result file is written in to result_evaluated.xls
- **eval.py**:
  - This py file computes perplexity and cosin similarity score for evaluation. 
- **generate.py**
  - This py file is from homework 1. This is a helper file for generating text.
- **models.py**
  - This py file is our main file. It can be used for training our language model. It can also be used for loading pre-trained weights and generate csv files containg tweets generated by pre-trained model into generated_text folder.
  - For transfer learning, please download file from https://nlp.stanford.edu/projects/glove/, and move glove.twitter.27B.25d.txt to the project folder.
  - After training, it documents it's loss, generated text and parameters into result. csv.
- **result.csv**
  - This csv contains result generated by models.py's training. 
- **parse_tweets.py**
  - This py file parses tweets.
- **twitter_api.py**
  - This py file scraped tweets using twitter api. 
- **generate_text_script.sh**
  - This sh script file helps generate all the csv files in generated_text. 
  - This script file goes through all* the models. To run the subset of the models, run generate_text_script_small.sh please. 
- **generate_text_script_small.sh**
  - Same as above, but this file only runs the models already contained in the trained_model folder.(All the LSTM models) 
- **run.sh**
  - This sh script file helps run all the training. 
- **ALL_AOC_TWEETS.txt**
  - This txt file contains 2300 scraped Congresswoman Alexandria Ocasio-Cortez's tweets.
- **AOC_500_tweets.txt**
  - This txt file contains 500 scraped Congresswoman Alexandria Ocasio-Cortez's tweets.
- **ALL_TRUMP_TWEETS.txt**
  - This txt file contains scraped President Trump's tweets.
- **TRUMP_500_TWEETS.txt**
  - This txt file contains 500 scraped President Trump's tweets.
- **TRUMP_2600_TWEETS.txt**
  - This txt file contains 2600 scraped President Trump's tweets.
- **evaluation_result.csv**
  - This csv file contains evaluation results generated by binary_classifier.py of all the csv files inside of generated_text folder.


 
  

  
  
  
## Training Model
Run the following command: `models.py --type train  -tf PARSED_TWEETS_FILE_PATH -wf WEIGHTS_FILE_PATH`
Examples of running commands can be found in run.sh which is the script file we used for training. 
There are a number of optional arguments that you can set: 
- -e: Number of epochs (default is 50)
- -do: Dropout (default is 0.3)
- -ea: Early stopping parameter, to have no early stopping use -1 (default is 0.1)
- -em: Embedding size (default is 100)
- -hs: Number of nodes in dense hidden layers (default is 100)

## Building Model from Pre-Cached Weights 
Run the following command: `models.py --type load  -tf PARSED_TWEETS_FILE_PATH -wf WEIGHTS_FILE_PATH`
Examples of running commands can be found in generate_text_script.sh which is the script file we used for loading pre-trained models.
In this case, since we only uploaded lstm models to github due to the file size restriction, please run generate_text_script_small.sh. By running this script, the models.py file will load the models and generate 100 tweets for each of the model. The csv file containing these tweets will be in generated_text folder. 

## Scraping from Twitter + Using our Parser
There are two steps you'll have to do to get tweets into a suitable format for our model. 
1. Scrape tweets
Setup a config.py using the format from config_format.py. Then, run the following command: `python3 twitter_api.py --extract USER_NAME`

2. Parse tweets
Run the following command: `python3 parse_tweets.py --parse RAW_TWEET_FILE_PATH`
This sets up the tweets in a list of lists. Each tweet's list is of the format [TWEET_ID, TWEET_DATE, TWEET_TIME, TWEET_TEXT]. 

In our parsing, we made some decisions for how to tokenize tweets, for example:

- We added whitespace around most punctuation marks so that they are their own tokens. However, we left conjunctions in tact (it's, I'd.)
- We removed all links (which represent either quote tweets or images.) We replaced these with a MEDIA tag (defaults to "<MEDIA>".)
- We removed all "\n" characters within a tweet and instead added a newline indicator "<NL>" to preseve structure in the recreations. 
- We left in emojis. 


## Transfer Learning

Navigate to https://nlp.stanford.edu/projects/glove/ and download a pretrained Twitter embedding (note: file size is large).

## Binary Classifier Evaluation
### Reference https://stackabuse.com/text-classification-with-python-and-scikit-learn/
Here we trained a binary classifier by using sklearn's MultinomialNB as our classifier and both Trump and Alexandria Ocasio-Cortez's actual tweets as our training set. We first converted the tweets into numerical features using bag of words model. Then we transformed the features using TFIDF which solves the issue that the bag of words model does not take into account that certain words might have high occurance in other texts. We achieved a 0.92 accuracy. We saved the model and CountVectorizer for faster evaluation. (Warning: different version of sklearn might result in failure of loading pickle file. In this case, delete the pickle file and redo the training.)

Then the script will loop all the files in the generated text folder, and evaluate all the result files by classifying whether the tweet is AOC's or Trump's. An accuracy score is calculated by number of correct classifications divided by size of genertated text file,in this case 100. All the results is automatically compiled into evaluation_result.csv. 

Lastly, the script with loop all the generated tweets in the result file, since some of the weights cannot be loaded properly. A score is calculated and added to the result dataframe. Then it outputs a result_evaluated.xlsx since csv file messes up the row for some reason. 

## Perplexity & Similarity Evaluation
To run the perplexity and similarity, use `eval.py` and see the `run_eval.sh` script for formatting. 
