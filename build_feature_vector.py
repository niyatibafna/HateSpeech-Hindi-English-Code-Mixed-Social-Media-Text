#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
import numpy as np
import operator
import pickle
import preprocessing as pre
import re
import string
import json
from tqdm import tqdm
from features_count import *
global char_n_grams_index, word_n_grams_index, hate_words_index, feature_vector_file

feature_vector_file = "fv_cgrams.json"


import os

def AddEmoticonFeatures(feature_vector, happy_emoticon, sad_emoticon,
						disgust_emoticon, anger_emoticon, fear_emoticon,
						surprise_emoticon):
	count_happy = count_sad = count_disgust = count_anger = count_fear = count_surprise = 0
	#print "AddEmoticonFeatures Called"
	for emoticon in pre.all_emoticons:
		if emoticon in happy_emoticon:
			count_happy = count_happy + 1
		elif emoticon in sad_emoticon:
			count_sad = count_sad + 1
		elif emoticon in anger_emoticon:
			count_anger = count_anger + 1
		'''
		elif emoticon in disgust_emoticon:
			count_disgust = count_disgust + 1

		elif emoticon in fear_emoticon:
			count_fear = count_fear + 1
		elif emoticon in surprise_emoticon:
			count_surprise = count_surprise + 1
		'''
	feature_vector.append(count_happy)
	feature_vector.append(count_sad)
	#feature_vector.append(count_disgust)
	feature_vector.append(count_anger)
	#feature_vector.append(count_fear)
	#feature_vector.append(count_surprise)

	return feature_vector

def AddCharNGramFeatures(feature_vector, char_n_grams_index, char_n_grams):
	char_features = [0]*len(char_n_grams_index)
	#print "AddCharNGramFeatures Called"
	for char_gram in char_n_grams:
		if char_gram in char_n_grams_index:
			char_features[char_n_grams_index[char_gram]] = 1
	feature_vector.extend(char_features)
	return feature_vector

def AddWordNGramFeatures(feature_vector, word_n_grams_index, word_n_grams):
	word_features = [0]*len(word_n_grams_index)
	#print "AddWordNGramFeatures Called"
	for word_gram in word_n_grams:
		if word_gram in word_n_grams_index:
			word_features[word_n_grams_index[word_gram]] = 1
	feature_vector.extend(word_features)
	return feature_vector

def AddHateWordsFeature(feature_vector, hate_words_index, tweet_hate_words):
	hate_feature = [0]*len(hate_words_index)
	#print len(hate_feature)
	for hate_word in tweet_hate_words:
		if hate_word in hate_words_index:
			#print hate_word
			hate_feature[hate_words_index[hate_word]] = 1
			#print hate_words_index[hate_word]
	feature_vector.extend(hate_feature)
	return feature_vector

def AddPunctuationMarksFeature(feature_vector, punctuations_marks_count):
	for punctuation in pre.punctuations_marks:
		if punctuation in punctuations_marks_count:
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	feature_vector.append(len(punctuations_marks_count))

	return feature_vector

def AddRepetitiveWordsFeature(feature_vector, repetitive_words):
	#print "AddRepetitiveWordsFeature Called"
	if len(repetitive_words) > 0:
		feature_vector.append(1)
	else:
		feature_vector.append(0)
	#feature_vector.append(len(repetitive_words))

	return feature_vector

def AddUpperCaseWordsFeature(feature_vector, upper_case_words):
	if len(upper_case_words) > 0:
		feature_vector.append(1)
	else:
		feature_vector.append(0)
	#feature_vector.append(len(upper_case_words))

	return feature_vector

def AddIntensifersFeature(feature_vector, intensifiers):
	if len(intensifiers) > 0:
		feature_vector.append(1)
	else:
		feature_vector.append(0)
	#feature_vector.append(len(intensifiers))

	return feature_vector

def AddNegationsFeature(feature_vector, negations):
	if len(negations) > 0:
		feature_vector.append(1)
	else:
		feature_vector.append(0)
	#feature_vector.append(len(negations))
	return feature_vector


def BuildFeatureVectorForTweet(tweet, mode = ["cgrams", "wgrams"]):

	#print "BuildFeatureVectorForTweet Called"
	global char_n_grams_index, word_n_grams_index, hate_words_index
	happy, sad, anger, fear, surprise, disgust, hashtags, usernames, \
	urls, punctuations_marks_count, repetitive_words, char_n_grams, \
	word_n_grams, upper_case_words, intensifiers, negations, tweet_hate_words = pre.PreProcessing(tweet)
	#print tweet_hate_words
	feature_vector = []
	#print char_n_grams_index
	#print word_n_grams_index
	if "all" in mode:
		feature_vector = AddEmoticonFeatures(feature_vector, happy, sad, disgust, anger, fear, surprise)
		feature_vector = AddRepetitiveWordsFeature(feature_vector, repetitive_words)
		feature_vector = AddPunctuationMarksFeature(feature_vector, punctuations_marks_count)
		feature_vector = AddUpperCaseWordsFeature(feature_vector, upper_case_words)
		feature_vector = AddIntensifersFeature(feature_vector, intensifiers)
		feature_vector = AddNegationsFeature(feature_vector, negations)

	if "hatewords" in mode:
		feature_vector = AddHateWordsFeature(feature_vector, hate_words_index, tweet_hate_words)
	if "wgrams" in mode:
		feature_vector = AddWordNGramFeatures(feature_vector, word_n_grams_index, word_n_grams)
	if "cgrams" in mode:
		feature_vector = AddCharNGramFeatures(feature_vector, char_n_grams_index, char_n_grams)

	return feature_vector


def GetIndexes(tweet_mapping, fpath):
	global char_n_grams_index, word_n_grams_index, hate_words_index
	if not os.path.exists(fpath):
		print("Pickling indexes")
		CreatePickleFile(tweet_mapping, fpath)

	file = open(fpath, "rb")
	data = []
	# If file reaches the EOL while reading this will reset and the reading
	# will start from beginning
	file.seek(0)

	for i in range(pickle.load(file)):
		data.append(pickle.load(file))

	char_n_grams_index, word_n_grams_index, hate_words_index = data

	file.close()



def FeatureVectorDictionary(tweet_mapping, index_fpath, mode = ["cgrams"]):
	global char_n_grams_index, word_n_grams_index, hate_words_index
	print("Building FeatureVectorDictionary")
	GetIndexes(tweet_mapping, index_fpath)
	feature_vector_dict = {}
	for key, tweet in tqdm(list(tweet_mapping.items())):
		feature_vector_dict[key] = BuildFeatureVectorForTweet(tweet, mode)
	return feature_vector_dict


def LoadFeatureVectorDictionary(filename):
	print("Loading FeatureVectorDictionary")
	with open("feature_vector_dicts/"+filename, "r") as dict_file:
		return {int(k): val for k, val in json.load(dict_file).items()}

def SaveFeatureVectorDictionary(feature_vector_dictionary, filename):
	print("Saving FeatureVectorDictionary")
	with open("feature_vector_dicts/"+filename, "w") as dict_file:
		json.dump(feature_vector_dictionary, dict_file)


def TrainingData(id_tweet_map, id_class_map, index_fpath = "indexes/indexes_hasoc.pkl", mode = ["cgrams"], req_feature_vector_file="fv_cgrams.json"):
	global feature_vector_file
	feature_vector_file = req_feature_vector_file


	#print "TrainingData Called"
	if not os.path.exists("feature_vector_dicts/"+feature_vector_file):

		feature_vector_dict = FeatureVectorDictionary(id_tweet_map, index_fpath, mode)
		# SaveFeatureVectorDictionary(feature_vector_dict, feature_vector_file)

	else:
		feature_vector_dict = LoadFeatureVectorDictionary(feature_vector_file)
	#print "Done"
	tweet_feature_vector = []
	tweet_class = []
	for key, val in tqdm(list(feature_vector_dict.items())):
		# print(key, val)
		# X = np.expand_dims(np.asarray(feature_vector_dict[key]), axis=0)
		# Y = np.expand_dims(np.asarray(id_class_map[key]), axis=0)
		# if tweet_feature_vector is None:
		# 	tweet_feature_vector = X
		# 	tweet_class = Y
		# else:
		# 	tweet_feature_vector = np.append(tweet_feature_vector, X, axis=0)
		# 	tweet_class = np.append(tweet_class, Y, axis=0)
		tweet_feature_vector.append(feature_vector_dict[key])
		tweet_class.append(id_class_map[key])

	return tweet_feature_vector, tweet_class


def TestData(id_tweet_map, index_fpath = "indexes/indexes_hasoc.pkl", mode = ["cgrams"]):

	tweet_feature_vector = []

	feature_vector_dict = FeatureVectorDictionary(id_tweet_map, index_fpath, mode)

	tweet_feature_vector = []
	tweet_class = []
	# for key, val in feature_vector_dict.items():
	# 	if not tweet_feature_vector:
	# 		tweet_feature_vector = np.unsqueeze(np.asarray(feature_vector_dict[key]), axis=0)
	# 	else:
	# 		tweet_feature_vector = np.append(tweet_feature_vector, feature_vector_dict[key], axis=0)
	for key in tqdm(sorted(list(feature_vector_dict.keys()))):
		tweet_feature_vector.append(feature_vector_dict[key])

	return tweet_feature_vector
