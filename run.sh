python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 100 -wf "1.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea -1 -em 100 -hs 100 -wf "1s.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea 0.1 -em 100 -hs 250 -wf "2.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea -1 -em 100 -hs 250 -wf "2s.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea 0.1 -em 250 -hs 500 -wf "3.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 50 -do 0.3 -ea 0.1 -em 500 -hs 500 -wf "4.h5"
python models.py -tf ALL_AOC_TWEETS.txt -t "train" -e 100 -do 0.3 -ea 0.1 -em 100 -hs 250 -wf "5.h5"

