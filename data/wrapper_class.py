import numpy as np
import random as rand
import sys
import re
import string
from itertools import groupby
import json

class WrapperClass:
    def __init__(self, dataset='train'):
        self.crossword_counter = 0
        self.dictionary_counter = 0
        crossword_file = None
        dict_file= None
        if dataset == 'train':
            crossword_file = 'train.txt'
            dict_file = 'dict_train.txt'
        elif dataset == 'dev':
            crossword_file = 'dev.txt'
            dict_file = 'dict_dev.txt'
        elif dataset == 'test':
            crossword_file = 'test.txt'
            dict_file = 'dict_test.txt'
        with open(crossword_file, mode='r') as input_file:
            crossword_lines = list(input_file)
        rand.shuffle(crossword_lines)
        self.crossword_lines = crossword_lines
        self.num_crossword_examples = len(crossword_lines)

        with open(dict_file, mode='r') as input_file:
            dictionary_lines = list(input_file)
        rand.shuffle(dictionary_lines)
        self.dictionary_lines = dictionary_lines
        self.num_dictionary_examples = len(dictionary_lines)

    def get_crossword_batch(self,  dimensions=64):
        lines_batch = self.crossword_lines[self.crossword_counter:self.crossword_counter+dimensions]
        self.crossword_counter = self.crossword_counter + dimensions
        if self.crossword_counter > len(self.crossword_lines):
            rand.shuffle(self.crossword_lines)
            self.crossword_counter = 0
        if len(lines_batch) == 0:
            lines_batch = self.crossword_lines[self.crossword_counter:self.crossword_counter+dimensions]
            self.crossword_counter= self.crossword_counter+ dimensions

        tuple_list = []
        for line in lines_batch:
            line = line.replace(';',' ')
            words = line.split()
            if len(words) > 0:
                cur_word = words[-1]
                # if cur_word in vocabulary_dict.keys():
                tuple_list.append((cur_word, words[:-1]))

        return tuple_list

    def get_dictionary_batch(self, dimensions=64):
        lines_batch = self.dictionary_lines[self.dictionary_counter:self.dictionary_counter+dimensions]
        self.dictionary_counter = self.dictionary_counter + dimensions
        if self.dictionary_counter > len(self.dictionary_lines):
            rand.shuffle(self.dictionary_lines)
            self.dictionary_counter = 0
        if len(lines_batch) == 0:
            lines_batch = self.dictionary_lines[self.dictionary_counter:self.dictionary_counter+dimensions]
            self.dictionary_counter = self.dictionary_counter + dimensions
            
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

