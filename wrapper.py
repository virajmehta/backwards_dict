import numpy as np
import random as rand
import sys
import re
import string
from itertools import groupby
import json

def get_crossword_batch(option, dimensions=64):
	if option == 'train':
		file_path = 'train.txt'
	if option == 'test':
		file_path = 'test.txt'
	if option == 'dev':
		file_path = 'dev.txt'

	with open(file_path, mode='r') as input_file:
		lines = list(input_file)
	rand.shuffle(lines)
	lines_batch = lines[0:dimensions]

	tuple_list = []
	for line in lines_batch:
		line = line.replace(';',' ')
		words = line.split()
		cur_word = words[-1]
		# if cur_word in vocabulary_dict.keys():
		tuple_list.append((cur_word, words[:-1]))

	return tuple_list

def get_dictionary_batch(option, dimensions=64):
	if option == 'train':
		file_path = 'train.txt'
	if option == 'test':
		file_path = 'test.txt'
	if option == 'dev':
		file_path = 'dev.txt'

	with open(file_path, mode='r') as input_file:
		lines = list(input_file)
	rand.shuffle(lines)
	lines_batch = lines[0:dimensions]

	tuple_list = []
	for line in lines_batch:
		while '  ' in line:
			line = line.replace('  ',' ')
		# line_words = line.split()
		# line = ' '.join(line_words)

		line = re.sub("[\(\[].*?[\)\]]", "", line)
		definitions = re.findall('\d+|\D+',line)
		definitions = [definition for definition in definitions if not definition.isdigit()]

		definitions_nopunct = []
		for definition in definitions:
			definition = ''.join(i for i in definition if i not in ('!',',','.','?',':',';'))
			definitions_nopunct.append(definition)

		tokenized_definitions = []
		for definition in definitions_nopunct:
			words = definition.split()
			tokenized_definitions.append(words)

		if len(tokenized_definitions) > 0 and len(tokenized_definitions[0]) > 0:
			cur_word = tokenized_definitions[0][0]
		
		if len(tokenized_definitions[0]) > 1:
			tokenized_definitions[0] = tokenized_definitions[0][1:]

		for definition in tokenized_definitions:
			if len(definition) >= 2:
				tuple_list.append((cur_word,definition))

	return tuple_list

def get_all_words(dictionary_path):
    all_words = {}
    index = 0
    with open(dictionary_path, mode='r') as input_file:
	    lines = list(input_file)
    for line in lines:
        words = line.split()
        if len(words) > 1:
            if words[0].lower() not in all_words:
                all_words[words[0].lower()] = index
                index += 1
    dict_file = open('vocabulary.txt','w')
    json.dump(all_words,dict_file)
    dict_file.close()

	# return vocabulary_dict





def main(argv):
	# dict_tuple_list = get_dictionary_batch(argv[2],64)
	# crossword_tuple_list = get_crossword_batch(argv[2],64,vocabulary)
	get_all_words('dictionary.txt')
	

if __name__ == "__main__":
	main(sys.argv)


