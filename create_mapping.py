import csv

text_id = 2
id_to_tweet_map = {}
tweet_to_id_map = {}
id_to_class_map = {}

def label_mapper(label):
	if label == "yes":
		return 1
	return 0



with open('../political_health/data/train/hate_speech.tsv') as dataset:
	for line in csv.reader(dataset, delimiter="\t"):

		class_name = []
		text = line[0]
		if len(line[1]) > 0:
			class_name = line[1].strip().lower()
			if class_name in ["n", "on"]:
				class_name = "no"
			assert class_name in ["yes", "no"]

		id_to_tweet_map[text_id] = text
		tweet_to_id_map[text] = text_id
		id_to_class_map[text_id] = label_mapper(class_name)

		#if len(emotions) == 0:
			#empty_emotions = empty_emotions + 1
			#print text_id, text
		text_id = text_id + 1



#print id_to_tweet_map[1454]
#print tweet_to_id_map['Live aane ke liye switch to #JIO services !!!']
#print id_to_class_map[1454]
