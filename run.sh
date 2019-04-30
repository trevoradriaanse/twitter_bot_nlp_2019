# 500 tweets Trump 

python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_500_trump_1.h5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_500_trump_2.h5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_500_trump_3.h5"

python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "lstm_500_trump_4.h5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "lstm_500_trump_5.h5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "lstm_500_trump_6.h5"


# 500 tweets Trump 

python3 models.py -tf AOC_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_500_aoc_1.h5"
python3 models.py -tf AOC_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_500_aoc_2.h5"
python3 models.py -tf AOC_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_500_aoc_3.h5"

python3 models.py -tf AOC_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "lstm_500_aoc_4.h5"
python3 models.py -tf AOC_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "lstm_500_aoc_5.h5"
python3 models.py -tf AOC_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "lstm_500_aoc_6.h5"




# 2600 tweets Trump  No LSTM 
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_2600_trump_1.h5"
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_2600_trump_2.h5"
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_2600_trump_3.h5"

# 2600 tweets AOC  No LSTM 
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_2600_aoc_1.h5"
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_2600_aoc_2.h5"
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_2600_aoc_3.h5"



