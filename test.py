from __future__ import division
import numpy as np
import tensorflow as tf

from random import randrange
from data.wrapper_class import WrapperClass
from lib.progbar import Progbar


def top10(config, embeddings, tokens, model):
    sentence = ''
    sess = tf.Session()
    with sess as session:
        saver = tf.train.Saver()
        saver.restore(session, model.config.saved_input)
        while True:
            print 'Input a clue or definition for the backwards dictionary:'
            try:
                sentence = raw_input()
            except EOFError:
                return
            tokens = sentence.split()
            input = []
            for word in tokens:
                try:
                    input.append(model.tokens[word.lower()])
                except:
                    pass
            length_batch = np.array([len(input)])
            for _ in range(config.max_length - len(input)):
                input.append(0)
            inputs_batch = np.array([input])
            input_shape = list(inputs_batch.shape)
            input_shape.append(1)
            inputs_batch1 = np.reshape(inputs_batch, input_shape)
            feed = model.create_feed_dict(inputs_batch1, length_batch=length_batch,
                                     dropout=1)
            logits = sess.run([model.pred], feed_dict=feed)[0][0]
            largestindices = np.argpartition(logits, -10)[-10:]
            if model.backwards is None:
                model.backwards = dict((v, k) for k, v in model.tokens.iteritems())
            top10words = [model.backwards[index] for index in largestindices]
            top10logits  = [logits[index] for index in largestindices]
            print 'top 10 guesses'
            for index, word in enumerate(top10words):
                print '{}, logit= {}'.format(word, top10logits[index])


def isCorrectChar(truth, prediction, index):
    if len(truth) != len(prediction):
        return false
    return prediction[index] == truth[index]

def isCorrectLength(truth, prediction):
    return len(truth) == len(prediction)


def eval_test(embeddings, tokens, model):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model.config.saved_input)
        test = WrapperClass('test')
        total_examples = 0.0
        num_correct= 0.0
        top_10_num_correct = 0.0
        top_10_length_correct = 0.0
        top_10_char_correct = 0.0
        top_50_length_correct = 0.0
        top_50_char_correct = 0.0
        for _ in range(int(test.num_crossword_examples / model.config.batch_size)):
            batch = test.get_crossword_batch(dimensions=model.config.batch_size)
            inputs = []
            labels = []
            lengths = []
            for example in batch:
                input = []
                for word in example[1][:40]:
                    try:
                        input.append(tokens[word.lower()])
                    except:
                        pass
                try:
                    labels.append(tokens[example[0].lower()])
                except:
                    continue
                length = len(input)
                for _ in range(model.config.max_length - length):
                    input.append(0)
                inputs.append(input)
                lengths.append(length)
            inputs_batch = np.array(inputs)
            input_shape = list(inputs_batch.shape)
            input_shape.append(1)
            inputs_batch1 = np.reshape(inputs_batch, input_shape)
            length_batch = np.array(lengths)
            feed = model.create_feed_dict(inputs_batch1, length_batch=lengths,
                                         dropout=1)
            logits = sess.run([model.pred], feed_dict=feed)[0]
            largest10indices = np.argpartition(logits, -10, axis=1)[:,-10:]
            largest50indices = np.argpartition(logits, -10, axis=1)[:,-10:]
            pred_labels = np.argmax(logits, axis=1)
            for _ in range(len(labels)):
                total_examples += 1
                if pred_labels[_] == labels[_]:
                    num_correct += 1
                if labels[_] in largest10indices[_,:]:
                    top_10_num_correct += 1
                truth =  tokens[labels[_]]
                char_index = randrange(len(truth))
                found_length = False
                for _, index in enumerate(largest50indices):
                    if isCorrectLength(truth, tokens[index]):
                        if index == labels[_] and not found_length:
                            if _ < 10:
                                top_10_length_correct += 1
                            top_50_length_correct += 1
                        found_length = True
                    if isCorrectChar(truth, tokens[index]):
                        if index == labels[_]:
                            if _ < 10:
                                top_10_char_correct += 1
                            top_50_char_correct += 1
                        break

        accuracy = num_correct / total_examples
        top_10_accuracy = top_10_num_correct / total_examples
        top_10_length_accuracy = top_10_length_correct / total_examples
        top_10_char_accuracy = top_10_char_correct / total_examples
        top_50_length_accuracy = top_50_length_correct / total_examples
        top_50_char_accuracy = top_50_char_correct / total_examples
        print 'test accuracy: ', accuracy
        print 'top 10 accuracy: ', top_10_accuracy
        print 'top 10 length accuracy: ', top_10_length_accuracy
        print 'top 10 char accuracy: ', top_10_char_accuracy
        print 'top 50 length accuracy: ', top_50_char_accuracy
        print 'top 50 char accuracy: ', top_50_char_accuracy

        return accuracy, top_10_accuracy
