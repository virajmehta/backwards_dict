import sys
from os import listdir
from os.path import isfile, join
import math
import random as rand
import glob
import json
import copy

#data/raw_xword_a

# path is 'data/'
def combine_files(data_path,output_path):
	files = [f for f in listdir(data_path) if isfile(join(data_path,f))]
	# files = glob.glob('data/*')
	# print files
	with open(output_path,'w') as output_file:
		for f in files:
			f = 'data/' + f
			with open(f,'rb') as input_file:
				output_file.write(input_file.read())

#path is 'all_words.txt'
def split_files(input_path,training_ratio,test_ratio,dev_ratio):
	with open(input_path, mode='r') as input_file:
		lines = list(input_file)
	rand.shuffle(lines)

	with open('vocabulary.txt', mode='r') as input_file:
		word_dict = json.load(input_file)

	all_words = []
	for word in word_dict:
		all_words.append((word.encode('ascii','ignore')).lower())
	# print all_words
	# print all_words

	# lines = lines[:10]

	real_lines = []

	for line in lines:
		# line_copy = copy.copy(line)
		line_copy = line.replace(';',' ')
		tokens = line_copy.split()
		# print tokens[-1]
		# test = tokens[-1] + '\n'
		# print test
		# print line
		# print line_copy
		# print tokens[-1]
		if len(tokens) > 1:
			if tokens[-1] in all_words:
			# print "YES"

				real_lines.append(line)

	train, test, dev = real_lines[:int(math.floor(len(real_lines)*training_ratio))],real_lines[int(math.ceil(len(real_lines)*training_ratio)):int(math.floor(len(real_lines)*(training_ratio+test_ratio)))],real_lines[int(math.ceil(len(real_lines)*(training_ratio+test_ratio))):int(math.floor(len(real_lines)*(training_ratio+test_ratio+dev_ratio)))]

	train_file = open('train.txt','w')
	for line in train:
		train_file.write(line)
	train_file.close()

	test_file = open('test.txt','w')
	for line in test:
		test_file.write(line)
	test_file.close()

	dev_file = open('dev.txt','w')
	for line in dev:
		dev_file.write(line)
	dev_file.close()

# def get_all_words(dictionary_path):
# 	all_words = []
# 	with open(dictionary_path, mode='r') as input_file:
# 		lines = list(input_file)
# 	for line in lines:
# 		words = line.split()
# 		if len(words) > 1:
# 			all_words.append(words[0])
# 	vocab = open('vocab.txt','w')
# 	for word in all_words:
# 		vocab.write(word+'\n')
# 	vocab.close()

	# vocabulary = dict(enumerate(all_words))
	# vocabulary_dict = dict((v,k) for k,v in vocabulary.iteritems())
	# dict_file = open('vocabulary.txt','w')
	# json.dump(vocabulary_dict,dict_file)
	# dict_file.close()
	# return all_words

def main():
	data_path = 'data/'
	output_path = 'combined_words.txt'
	input_path = 'dictionary.txt'
	combine_files(data_path,output_path)
	# all_words = get_all_words(input_path)
	split_files(output_path,.7,.2,.1)
	# get_all_words(input_path)

if __name__ == "__main__":
	main()


