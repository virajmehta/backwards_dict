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


logger = logging.getLogger("bag_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config(object):
    max_length=40
    embed_size=300
    n_features=1
    n_epochs=50
    batch_size=64
    dropout=0.5
    def __init__(self):
        #self.cell = args.cell

        self.vocab_size = None
        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}".format('bag', datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format('bag')
        self.log_output = self.output_path + "log"
        self.saved_input = '/home/toast/backwards_dict/results/bag/20170313_203006model.weights'




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

class BagModel(LSTMModel):

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        bag = tf.reduce_sum(x, 1)
        init = tf.cast(tf.transpose(tf.constant(self.pretrained_embeddings)), tf.float32)
        U = tf.get_variable('U', initializer=init)
        b = tf.get_variable('b', (self.config.vocab_size))
        pred = tf.matmul(bag, U) + b
        return pred

        

def main(config, embeddings, tokens):
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    # TODO: get examples
    graph = tf.Graph()
    with graph.as_default():
        logger.info("Building model...",)
        start = time.time()
        model = BagModel(config, embeddings, tokens)
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
            model = BagModel(config, embeddings, tokens)
            top10(config, embeddings, tokens, model)
    elif x.top10eval:
        graph = tf.Graph()
        with graph.as_default():
            model = BagModel(config, embeddings, tokens)
            eval_test(embeddings, tokens, model)
    else:
        main(config, embeddings, tokens)
