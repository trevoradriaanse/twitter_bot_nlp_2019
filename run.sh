# 500 tweets Trump 

python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea -1 -em 50 -hs 50 -wf "glove_500_trump_1.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea -1 -em 100 -hs 100 -wf "glove_500_trump_2.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea -1 -em 150 -hs 150 -wf "glove_500_trump_3.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea -1 -em 200 -hs 200 -wf "glove_500_trump_4.hdf5"

python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea -1 -em 50 -hs 50 -wf "lstm_500_trump_1.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea -1 -em 100 -hs 100 -wf "lstm_500_trump_2.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea -1 -em 150 -hs 150 -wf "lstm_500_trump_3.hdf5"
python3 models.py -tf TRUMP_500_tweets.txt -t "train" -g "regular" -e 50 -do 0.3 -ea -1 -em 200 -hs 200 -wf "lstm_500_trump_4.hdf5"

# 2600 tweets Trump  No LSTM 
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_2600_trump_1.h5"
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_2600_trump_2.h5"
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 150 -hs 150 -wf "glove_2600_trump_3.h5"
python3 models.py -tf TRUMP_2600_tweets.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_2600_trump_4_early.h5"

# 2600 tweets AOC  No LSTM 
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 50 -hs 50 -wf "glove_2600_aoc_1.hdf5"
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "glove_2600_aoc_2.hdf5"
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 150 -hs 150 -wf "glove_2600_aoc_3.hdf5"
python3 models.py -tf ALL_AOC_TWEETS.txt -t "train" -g "glove" -e 50 -do 0.3 -ea 0.1 -em 200 -hs 200 -wf "glove_2600_aoc_4_early.hdf5"



