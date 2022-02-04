#TESTING SCRIPT

from __future__ import division
import os
import numpy
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split  # Random split into training and test dataset.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from build_feature_vector import *
import sys
sys.path.append("../political_health/")
from data_reader import *

labels_exist = False

PATH_TO_CRAWLED_DATA = "../political_health/data/crawled"
INDEX_PATH = "indexes/indexes_cmsel_cgrams234.pkl"
MODEL_PATH = "models/svm_cmsel_wgrams_hwords.pkl"
KBEST_PATH = "models/selectkbest_cmsel_wgrams_hwords.pkl"
MODE = ["wgrams", "hatewords"]
PATH_TO_HASOC_DATA = "../political_health/data/hasoc"
# Load selectkbest_obj model
print("Loading SelectKBest Object")
with open(KBEST_PATH, "rb") as skbf:
	selectkbest_obj = pickle.load(skbf)

# Load SVM model
print("Loading SVM Model")
svm_fv_cgrams = pickle.load(open(MODEL_PATH, "rb"))

# GET DATA
tweet_reader = TweetReader(PATH_TO_CRAWLED_DATA)
id_tweet_map = tweet_reader.reader([2011], ["eid"])

id_tweet_map = {key:val for key,val in id_tweet_map.items() if key<5000}
# id_class_map = {key:val for key,val in id_class_map.items() if key<102}

# id_tweet_map = {0:"I want to drink water"}

tweets_list = [id_tweet_map[key] for key in sorted(list(id_tweet_map.keys()))]
if labels_exist:
	hasoc = HasocReader(PATH_TO_HASOC_DATA)
	id_tweet_map, id_class_map = hasoc.reader(dict(), dict())

	# id_tweet_map = create_id_tweet_map()
	# id_class_map = create_id_class_map()

	class_list = [id_class_map[key] for key in sorted(list(id_class_map.keys()))]


# Prepare feature vectors
X = TestData(id_tweet_map, index_fpath = INDEX_PATH, mode = MODE)
# Convert list into a array
print("Number of features: ", len(X[0]))

X = numpy.asarray(X)
X = selectkbest_obj.transform(X)

# print(X)

print("Running SVM Model")
predictions = svm_fv_cgrams.predict(X)
# print(predictions)
if labels_exist:
	prec_score = precision_score(class_list, predictions)
	rec_score = recall_score(class_list, predictions)
	print("Precision: ", prec_score)
	print("Recal: ", rec_score)
	print("Accuracy : " , accuracy_score(class_list, predictions))


for idx, elem in enumerate(predictions):
	if elem == 1:
		print(idx, tweets_list[idx])
		print("Pred: ", elem)
		print("\n\n")

hate_tweets = len([x for x in predictions if x == 1])
print("Number of tweets labelled as hate speech: {} ".format(hate_tweets))
print("Percentage: {}".format(hate_tweets/len(id_tweet_map)))
