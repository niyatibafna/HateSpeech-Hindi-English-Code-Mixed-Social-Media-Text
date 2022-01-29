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


PATH_TO_CRAWLED_DATA = "../political_health/data/crawled"
INDEX_PATH = "indexes/indexes_cm_cgrams234.pkl"
MODEL_PATH = "models/svm_cm_cgrams234.pkl"
KBEST_PATH = "models/selectkbest_cm_cgrams234.pkl"

print("Loading SelectKBest Object")
with open(KBEST_PATH, "rb") as skbf:
	selectkbest_obj = pickle.load(skbf)


print("Loading SVM Model")
svm_fv_cgrams = pickle.load(open(MODEL_PATH, "rb"))


# tweet_reader = TweetReader(PATH_TO_CRAWLED_DATA)
# id_tweet_map = tweet_reader.reader([2010], ["musalman"])
#

# id_tweet_map = create_id_tweet_map()

# tweets_list = [id_tweet_map[key] for key in sorted(list(id_tweet_map.keys())) if key==221]

# id_tweet_map = {key:val for key,val in id_tweet_map.items() if key==221}
# print(id_tweet_map)

id_tweet_map = {0:"Kia ho ager teray rukhsaar ko hum choomtay hein; Jo musalmaan hein wo Quraan ko sanam choomtay hein! (Bahadur Shah Zafar)"}
tweets_list = [id_tweet_map[key] for key in sorted(list(id_tweet_map.keys()))]


X = TestData(id_tweet_map, index_fpath = INDEX_PATH, mode = ["cgrams"])
# Convert list into a array
X = numpy.asarray(X)
X = selectkbest_obj.transform(X)

# print(X)

print("Running SVM Model")
predictions = svm_fv_cgrams.predict(X)
print(predictions)
# print( "Accuracy : " , round(accuracy/10, 3))


for idx, elem in enumerate(predictions):
    if elem == 1:
        print(idx, tweets_list[idx])
        print("\n\n")
