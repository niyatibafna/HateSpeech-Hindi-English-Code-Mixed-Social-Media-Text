#!usr/bin/env python3

from run_svm import *
from collections import defaultdict
import itertools
import json


WORDLIST_DIR = "../political_health/wordlists/"
DATA_GENRE = "political_controversies"
WORDLIST = DATA_GENRE + ".txt"
YEARS = range(2019, 2021)
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
    outpath_hate = OUTPUT_DIR + keyword + ".txt"
    total, hate_percent = run_svm_all(outpath_hate, years = [year], keywords = [keyword.split()[0]], use_hasoc = use_hasoc)
    results_dict[year][keyword] = {"Total tweets":total, "Percent hate":hate_percent}
    print("Total tweets: {}".format(total))
    print("Number of tweets labelled as hate speech: {} ".format(total*hate_percent))
    print("Percentage: {}".format(hate_percent))

print(results_dict)

with open(OUTPUT_OVERVIEW_DIR+OUTPUT, "w") as f:
    json.dump(results_dict, f, indent = 2)
