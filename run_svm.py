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
from format_data import *
import sys
sys.path.append("../political_health/")
from hasoc_reader import *

PATH_TO_HASOC_DATA = "../political_health/data/hasoc"
INDEX_PATH = "indexes/indexes_cm_cgrams234.pkl"
FV_FILE = "fv_cm_wgrams_hwords.json"
MODEL_PATH = "models/svm_cm_wgrams_hwords.pkl"
KBEST_PATH = "models/selectkbest_cm_wgrams_hwords.pkl"
MODE = ["wgrams", "hatewords"]

# Get data, HASOC data
id_tweet_map = create_id_tweet_map()
id_class_map = create_id_class_map()

print("Length of training data: ", len(id_tweet_map))

hasoc = HasocReader(PATH_TO_HASOC_DATA)
id_tweet_map, id_class_map = hasoc.reader(id_tweet_map, id_class_map)

print("Length of all training data (including HASOC): ", len(id_tweet_map))
assert len(id_tweet_map) == len(id_class_map)

# Prepare feature vectors
X, Y = TrainingData(id_tweet_map, id_class_map, index_fpath = INDEX_PATH, mode = MODE, req_feature_vector_file = FV_FILE)

# Convert list into a array
print("Features shape: ", len(X[0]))
X = numpy.asarray(X)
Y = numpy.asarray(Y)

# Data transformation
print("Selecting K best:")
selectkbest_obj = SelectKBest(chi2, k=1200).fit(X,Y)
# with open(KBEST_PATH, "rb") as skbf:
# 	selectkbest_obj = pickle.load(skbf)

X = selectkbest_obj.transform(X)

print("Saving K best model")
with open(KBEST_PATH, "wb") as skbf:
	pickle.dump(selectkbest_obj, skbf)

print("Running SVM")
clf = svm.SVC(kernel = 'rbf', C=10)
clf.fit(X, Y.ravel())
with open(MODEL_PATH, "wb") as m_file:
	pickle.dump(clf, m_file)


# Training with KFold accuracy
kf = KFold(n_splits=10)

fold = 0
accuracy = 0
for train_idx, test_idx in kf.split(X):
		fold = fold + 1
		X_train, X_test = X[train_idx], X[test_idx]
		Y_train, Y_test = Y[train_idx], Y[test_idx]
		clf = svm.SVC(kernel = 'rbf', C=10)
		clf.fit(X_train, Y_train.ravel())
		predictions = clf.predict(X_test)
		score = accuracy_score(Y_test, predictions)
		accuracy = accuracy + score
		print("Score for fold %d: %.3f" %(fold, score))


print( "Accuracy : " , round(accuracy/10, 3))
