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

print("Loading SelectKBest Object")
with open("models/selectkbest_cgrams.pkl", "rb") as skbf:
	selectkbest_obj = pickle.load(skbf)


print("Loading SVM Model")
svm_fv_cgrams = pickle.load(open("models/svm_fv_cgrams.pkl", "rb"))


tweet_reader = TweetReader("../political_health/data/crawled")
id_tweet_map = tweet_reader.reader([2011], ["eid"])

id_tweet_map = {key:val for key,val in id_tweet_map.items()}

X = TestData(id_tweet_map)
# Convert list into a array
X = numpy.asarray(X)
X = selectkbest_obj.transform(X)

print("Running SVM Model")
predictions = svm_fv_cgrams.predict(X)
print(predictions)
# print( "Accuracy : " , round(accuracy/10, 3))


for idx, elem in enumerate(predictions):
    if elem == "yes":
        print(id_tweet_map[idx])
        print("\n\n")
