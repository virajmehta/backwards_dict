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
from lstm_model import LSTMModel
from lib.glove import loadWordVectors
from lib.progbar import Progbar
from data.wrapper_class import WrapperClass
from test import top10, eval_test


logger = logging.getLogger("bidir")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config(object):
    max_length=40
    embed_size=50
    lstm_dimension=200
    n_features=1
    n_epochs=200
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
        self.saved_input = '/Users/virajmehta/Projects/backwards_dict/scr/bag/20170313_203006model.weights'




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

class BidirLSTMModel(LSTMModel):

    def add_prediction_op(self):
        num_layers = 3
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        with tf.name_scope('fwcell'):
            fw_cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        with tf.name_scope('bwcell'):
            bw_cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        with tf.name_scope('fwmulticell'):
            fw_multicell = tf.contrib.rnn.MultiRNNCell([fw_cell]*num_layers)
        with tf.name_scope('bwmulticell'):
            bw_multicell = tf.contrib.rnn.MultiRNNCell([bw_cell]*num_layers)
        with tf.name_scope('rnn'):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, x, dtype=tf.float32, sequence_length=self.length_placeholder)
            tf.summary.histogram('outputs', outputs)
        fw_state, bw_state = output_states
        fw_state_last = fw_state[1].c
        bw_state_last = bw_state[1].c
        print fw_state_last.get_shape()
        print fw_state_last
        concat_states = tf.concat([fw_state_last, bw_state_last], 1)
        with tf.name_scope('U'):
            U = tf.get_variable('U', (self.config.lstm_dimension, self.config.vocab_size),
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('U', U)
        with tf.name_scope('b2'):
            b2 = tf.get_variable('b2', (self.config.vocab_size))
            tf.summary.histogram('b2', b2)

        with tf.name_scope('W'):
            W = tf.get_variable("W",(self.config.vocab_size,2*self.config.vocab_size),initializer=tf.contrib.layers.xavier_initializer)
            tf.summary.histogram('W', W)


        # W: 0 dimension of forward state, 0 dimension of concat state (batch size by 2 batch size)

        with tf.name_scope('h'):
            h = tf.nn.relu(tf.matmul(W,concat_states) + b2)
            tf.summary.histogram('h', h)
        # CONFUSED ABOUT W DIMENSIONS
        #W = tf.Variable(initializer((Config.n_features * Config.embed_size, Config.vocab_size)))
        #b1 = tf.Variable(tf.zeros([Config.vocab_size]))

        #h = tf.nn.softmax(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)

        with tf.name_scope('pred'):
            pred = tf.matmul(h_drop, U) + b2
            tf.summary.histogram('pred', pred)
        return pred



def main(config, embeddings, tokens):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true')
    parser.add_argument('--top10eval', action='store_true')
    x = parser.parse_args()
    config = Config()
    embeddings, tokens = loadWordVectors()
    config.embed_size = embeddings.shape[1]
    config.vocab_size = len(tokens)
    if x.t:
        graph = tf.Graph()
        with graph.as_default():
            model = LSTMModel(config, embeddings, tokens)
            top10(config, embeddings, tokens, model)
    elif x.top10eval:
        graph = tf.Graph()
        with graph.as_default():
            model = LSTMModel(config, embeddings, tokens)
            eval_test(embeddings, tokens, model)
    else:
        main(config, embeddings, tokens)
