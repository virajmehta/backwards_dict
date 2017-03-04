import sys
from os import listdir
from os.path import isfile, join
import math
import random as rand
import glob

#data/raw_xword_a

# path is 'data/'
def combine_files(data_path,output_path):
	files = [f for f in listdir(data_path) if isfile(join(data_path,f))]
	# files = glob.glob('data/*')
	print files
	with open(output_path,'w') as output_file:
		for f in files:
			f = 'data/' + f
			print f
			with open(f,'rb') as input_file:
				output_file.write(input_file.read())

#path is 'all_words.txt'
def split_files(input_path,training_ratio,test_ratio,dev_ratio):
	with open(input_path, mode='r') as input_file:
		lines = list(input_file)
	rand.shuffle(lines)

	train, test, dev = lines[:int(math.floor(len(lines)*training_ratio))],lines[int(math.ceil(len(lines)*training_ratio)):int(math.floor(len(lines)*(training_ratio+test_ratio)))],lines[int(math.ceil(len(lines)*(training_ratio+test_ratio))):int(math.floor(len(lines)*(training_ratio+test_ratio+dev_ratio)))]

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

def main():
	print "HELLO"
	data_path = 'data/'
	output_path = 'all_words.txt'
	combine_files(data_path,output_path)
	split_files(output_path,.7,.2,.1)

if __name__ == "__main__":
	main()


