from __future__ import absolute_import
from __future__ import division


import argparse
import logging
import sys
import time
from datetime import datetime
import os

import numpy as np
import tensorflow as tf
from models.model import Model
from lstm_model import LSTMModel
from lib.glove import loadWordVectors
from lib.progbar import Progbar
from data.wrapper_class import WrapperClass


logger = logging.getLogger("stacked_lstm_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config(object):
    max_length=40
    embed_size=50
    lstm_dimension=200
    n_features=1
    n_epochs=20
    batch_size=64
    dropout=0.5
    def __init__(self):
        #self.cell = args.cell

        self.vocab_size = None
        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}".format('stacked', datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format('stacked')
        self.log_output = self.output_path + "log"
        self.summary_index = 0
        self.summary_path = self.output_path + 'summary'




def pad_sequences(data, max_length):
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = -1 # corresponds to the 'O' tag

    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)
            mask = None
            sentence_ = None
            labels_ = None
            if len(sentence) > max_length:
                sentence_ = sentence[:max_length]
                labels_ = labels[:max_length]
                mask = [True] * max_length
            else:
                sentence_ = sentence[:]
                labels_ = labels[:]
                for _ in range(max_length - len(sentence)):
                    sentence_.append(zero_vector)
                    labels_.append(zero_label)
                mask = [True] * len(sentence)
                mask.extend([False] * (max_length - len(sentence)))
            ret.append((sentence_, labels_, mask))
        ### END YOUR CODE ###
    return ret

class StackedLSTMModel(LSTMModel):

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell]*2)
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, sequence_length=self.length_placeholder)
        with tf.name_scope('U'):
            U = tf.get_variable('U', (self.config.lstm_dimension, self.config.embed_size),
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('U',U)
        with tf.name_scope('b1'):
            b1 = tf.get_variable('b1',(self.config.embed_size))
            tf.summary.histogram('b1',b1)
        init = tf.cast(tf.transpose(tf.constant(self.pretrained_embeddings)), tf.float32)
        with tf.name_scope('W'):
            W = tf.get_variable('W', initializer=init)
            tf.summary.histogram('W',W)
        with tf.name_scope('b2'):
            b2 = tf.get_variable('b2', (self.config.vocab_size))
            tf.summary.histogram('b2',b2)
        with tf.name_scope('h1'):
            h1 = tf.nn.relu(tf.matmul((state[1]).c,U) + b1)
            tf.summary.histogram('h1',h1)
        with tf.name_scope('h_drop'):
            h_drop = tf.nn.dropout(h1, dropout_rate)
            tf.summary.histogram('h_drop',h_drop)
        with tf.name_scope('pred'):
            pred = tf.matmul(h_drop, W) + b2
            tf.summary.histogram('pred',pred)
        return pred


def main():
    config = Config()
    embeddings, tokens = loadWordVectors()
    config.embed_size = embeddings.shape[1]
    config.vocab_size = len(tokens)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    # TODO: get examples
    graph = tf.Graph()
    with graph.as_default():
        logger.info("Building model...",)
        start = time.time()
        model = StackedLSTMModel(config, embeddings, tokens)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()

        writer = tf.summary.FileWriter('results/summary',graph)
        summary_op = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            model.fit(session, saver)

            summary = session.run(summary_op)
            writer.add_summary(summary,0)


            with open(model.config.conll_output, 'w') as f:
                write_conll(f, output)
            with open(model.config.eval_output, 'w') as f:
                for sentence, labels, predictions in output:
                    print_sentence(f, sentence, labels, predictions)

if __name__=='__main__':
    main()
