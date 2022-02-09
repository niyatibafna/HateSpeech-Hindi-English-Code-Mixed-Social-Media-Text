#!usr/bin/env python3

from run_svm import *
from collections import defaultdict
import itertools
import json
import os

#MODIFY
DATA_GENRE = "islamic_religious_terms" #islamic_religious_terms #political_controversies
YEARS = range(2010, 2021)

# NO NEED TO MODIFY!
WORDLIST_DIR = "../political_health/wordlists/"
WORDLIST = DATA_GENRE + ".txt"
OUTPUT_DIR = "../political_health/results/hate_speech/"
OUTPUT_OVERVIEW_DIR = "../political_health/results/overview/"
OUTPUT = "{}_{}-{}.json".format(DATA_GENRE, YEARS[0], YEARS[-1])

use_hasoc = False

def get_wordlist(filepath):
    with open(filepath, "r") as wl:
        return [word.strip() for word in wl.read().split("\n") if len(word)!=0]

wordlist = get_wordlist(WORDLIST_DIR+WORDLIST)
results_dict = defaultdict(lambda: dict())
for year, keyword in itertools.product(YEARS, wordlist):
    print("Processing: {}, {}".format(year, keyword))
    outpath_hate = OUTPUT_DIR + str(year) + "/"
    if not os.path.exists(outpath_hate):
        os.mkdir(outpath_hate)
    outpath_hate += keyword.split()[0] + ".txt"
    total, num_hate_tweets = run_svm_all(outpath_hate, years = [year], keywords = [keyword.split()[0]], use_hasoc = use_hasoc)
    if not total:
        print("Data not found!")
        continue
    hate_percent = num_hate_tweets/total
    results_dict[year][keyword] = {"Total tweets":total, "Number of hate tweets":num_hate_tweets, "Percent hate":hate_percent}
    print("Total tweets: {}".format(total))
    print("Number of tweets labelled as hate speech: {} ".format(num_hate_tweets))
    print("Percentage: {}".format(hate_percent))

print(results_dict)

with open(OUTPUT_OVERVIEW_DIR+OUTPUT, "w") as f:
    json.dump(results_dict, f, indent = 2)
