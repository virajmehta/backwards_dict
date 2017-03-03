import numpy as np
import random as rand
import sys
import re
import string
from itertools import groupby

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
	lines_batch = lines[0:64]

	tuple_list = []
	for line in lines_batch:
		line = line.replace(';',' ')
		words = line.split()
		cur_word = words[-1]
		tuple_list.append((cur_word, words[1:-1]))

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
	lines_batch = lines[0:64]

	tuple_list = []
	for line in lines_batch:
		while '  ' in line:
			line = line.replace('  ',' ')
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

			# while 'v' in words: words.remove('v')
			# while 'n' in words: words.remove('n')
			# while 'adj' in words: words.remove('adj')
			# while 'adv' in words: words.remove('adv')
			# while 'prep' in words: words.remove('prep')
			# while 'symb' in words: words.remove('symb')
			# while 'abbr' in words: words.remove('abbr')

			tokenized_definitions.append(words)

		if len(tokenized_definitions) > 0 and len(tokenized_definitions[0]) > 0:
			cur_word = tokenized_definitions[0][0]

		tokenized_definitions[0] = tokenized_definitions[0][1:]

		for definition in tokenized_definitions:
			if len(definition) >= 2:
				tuple_list.append((cur_word,definition))

	return tuple_list




def main(argv):
	if argv[1] == 'dictionary':
		tuple_list = get_dictionary_batch(argv[2],64)
		print tuple_list
	if argv[1] == 'crossword':
		tuple_list = get_crossword_batch(argv[2],64)
		print tuple_list
	

if __name__ == "__main__":
	main(sys.argv)


