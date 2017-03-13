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
from lib.glove import loadWordVectors
from lib.progbar import Progbar
from data.wrapper_class import WrapperClass


logger = logging.getLogger("bidir")
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
        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}".format('bidir', datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format('bidir')
        self.log_output = self.output_path + "log"
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

class BidirLSTMModel(Model):

    def add_prediction_op(self):
        num_layers = 2
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        fw_cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        bw_cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        fw_multicell = tf.contrib.rnn.MultiRNNCell(fw_cell*num_layers)
        bw_multicell = tf.contrib.rnn.MultiRNNCell(bw_cell*num_layers)
        outputs, fw_state, bw_state = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, x, dtype=tf.float32, sequence_length=self.length_placeholder)
        # print tf.shape(fw_state)
        # print tf.shape(bw_state)
        U = tf.get_variable('U', (self.config.lstm_dimension, self.config.vocab_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', (self.config.vocab_size))

        W = tf.get_variable("W",(self.config.vocab_size,2*self.config.vocab_size),initializer=tf.contrib.layers.xavier_initializer)

        concat_states = tf.concat(0, [fw_state.c, bw_state.c])

        # W: 0 dimension of forward state, 0 dimension of concat state (batch size by 2 batch size)

        h = tf.nn.relu(tf.matmul(W,concat_states) + b2)
        # CONFUSED ABOUT W DIMENSIONS
        #W = tf.Variable(initializer((Config.n_features * Config.embed_size, Config.vocab_size)))
        #b1 = tf.Variable(tf.zeros([Config.vocab_size]))

        #h = tf.nn.softmax(tf.matmul(x, W) + b1)
        #h_drop = tf.nn.dropout(h, dropout_rate)

        pred = tf.matmul(h, U) + b2
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
        model = BidirLSTMModel(config, embeddings, tokens)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            model.fit(session, saver)


            with open(model.config.conll_output, 'w') as f:
                write_conll(f, output)
            with open(model.config.eval_output, 'w') as f:
                for sentence, labels, predictions in output:
                    print_sentence(f, sentence, labels, predictions)

if __name__=='__main__':
    main()
