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


logger = logging.getLogger("lstm_model")
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
        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}".format('lstm', datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format('lstm')
        self.log_output = self.output_path + "log"




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

class LSTMModel(Model):

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, Config.max_length, Config.n_features),
                                                name='inputs')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                name='labels')
        self.length_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                name='lengths')
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, length_batch, labels_batch=None, dropout=1):
        feed_dict = {self.input_placeholder : inputs_batch,
                     self.dropout_placeholder : dropout, self.length_placeholder: length_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.Variable(self.pretrained_embeddings)
        wordvecs = tf.nn.embedding_lookup(all_embeddings, self.input_placeholder)
        embeddings = tf.reshape(wordvecs, (-1, Config.max_length, Config.n_features * Config.embed_size))
        return tf.cast(embeddings, tf.float32)

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        cell = tf.contrib.rnn.LSTMCell(Config.lstm_dimension)
        outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=self.length_placeholder)
        U = tf.get_variable('U', (self.config.lstm_dimension, self.config.vocab_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', (self.config.vocab_size))
        # CONFUSED ABOUT W DIMENSIONS
        #W = tf.Variable(initializer((Config.n_features * Config.embed_size, Config.vocab_size)))
        #b1 = tf.Variable(tf.zeros([Config.vocab_size]))

        #h = tf.nn.softmax(tf.matmul(x, W) + b1)
        #h_drop = tf.nn.dropout(h, dropout_rate)

        pred = tf.nn.softmax(tf.matmul(state.c, U) + b2)
        return pred

    def add_loss_op(self, pred):
        labels = self.labels_placeholder
        logits = pred
        ce= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(ce)
        return loss

    def add_training_op(self, loss):
        return tf.train.AdamOptimizer().minimize(loss)

    def train_on_batch(self, sess, batch):
        inputs = []
        labels = []
        lengths = []
        for example in batch:
            input = []
            for word in example[1][:40]:
                try:
                    input.append(self.tokens[word.lower()])
                except:
                    pass
            try:
                labels.append(self.tokens[example[0].lower()])
            except:
                continue
            length = len(input)
            for _ in range(self.config.max_length - length):
                input.append(0)
            inputs.append(input)
            lengths.append(length)
        inputs_batch = np.array(inputs)
        input_shape = list(inputs_batch.shape)
        input_shape.append(1)
        inputs_batch1 = np.reshape(inputs_batch, input_shape)
        labels_batch = np.array(labels)
        length_batch = np.array(lengths)
        feed = self.create_feed_dict(inputs_batch1, labels_batch=labels_batch, length_batch=lengths,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def run_epoch(self, sess):
        data = WrapperClass()
        prog = Progbar(target=1 + data.num_crossword_examples / self.config.batch_size)
        for _ in range(int(data.num_crossword_examples / self.config.batch_size)):
            batch = data.get_crossword_batch(dimensions=self.config.batch_size)
            dict_batch = data.get_dictionary_batch(dimensions=self.config.batch_size)

            loss = self.train_on_batch(sess, batch) #TODO
            loss += self.train_on_batch(sess, dict_batch)
            prog.update(_ + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        #TODO
        logger.info("Evaluating on development data")

        f1 = entity_scores[-1]
        return f1

    def fit(self, sess, saver):
        best_score = 0.

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score

    def __init__(self, config, pretrained_embeddings, tokens, report=None):
        self.input_placeholder = None
        self.labels_placeholder = None
        self.report = None
        self.mask_placeholder = None
        self.dropout_placeholder = None
        self.config = config
        self.pretrained_embeddings = pretrained_embeddings
        self.tokens = tokens
        self.build()

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
        model = LSTMModel(config, embeddings, tokens)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            model.fit(session, saver)

            output = model.output(session, dev_raw)
            sentences, labels, predictions = zip(*output)
            predictions = [[LBLS[l] for l in preds] for preds in predictions]
            output = zip(sentences, labels, predictions)

            with open(model.config.conll_output, 'w') as f:
                write_conll(f, output)
            with open(model.config.eval_output, 'w') as f:
                for sentence, labels, predictions in output:
                    print_sentence(f, sentence, labels, predictions)

if __name__=='__main__':
    main()
