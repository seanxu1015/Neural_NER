# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:11:24 2016

@author: sean
"""


import os, sys, time
import tensorflow as tf
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel
import numpy as np


class Config(object):
    embed_size = 50
    batch_size = 64
    label_size = 5
    hidden_size = 100
    max_epochs = 24
    early_stopping = 2
    dropout = 0.9
    lr = 0.001
    l2 = 0.001
    window_size = 3


class NERModel(LanguageModel):

    def load_data(self, debug=False):
        self.wv, word_to_num, num_to_word = ner.load_wv(
                'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
        tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
        self.num_to_tag = dict(enumerate(tagnames))
        tag_to_num = {v: k for k, v in self.num_to_tag.iteritems()}

        docs = du.load_dataset('data/ner/train')
        self.X_train, self.y_train = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_train = self.X_train[:1024]
            self.y_train = self.y_train[:1024]

        docs = du.load_dataset('data/ner/dev')
        self.X_dev, self.y_dev = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_dev = self.X_dev[:1024]
            self.y_dev = self.y_dev[:1024]

        docs = du.load_dataset('data/ner/test.masked')
        self.X_test, self.y_test = du.docs_to_windows(
            docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.window_size], name='Input')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=[None, self.config.label_size], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        feed_dict = {self.input_placeholder: input_batch}
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        return feed_dict

    def add_embedding(self):
        with tf.device('/gpu:0'):
            embedding = tf.get_variable(
                'Embedding', [len(self.wv), self.config.embed_size])
            window = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            window = tf.reshape(
                window, [-1, self.config.window_size * self.config.embed_size])
            return window

    def add_model(self, window):
        with tf.variable_scope('Layer1', initializer=xavier_weight_init()):
            W = tf.get_variable('W', [self.config.window_size * self.config.embed_size,
                                      self.config.hidden_size])
            b1 = tf.get_variable('b1', [self.config.hidden_size])
            h = tf.nn.tanh(tf.matmul(window, W) + b1)
            if self.config.l2:
                tf.add_to_collection('total_loss', 0.5*self.config.l2*tf.nn.l2_loss(W))

        with tf.variable_scope('Layer2', initializer=xavier_weight_init()):
            U = tf.get_variable('U', [self.config.hidden_size, self.config.label_size])
            b2 = tf.get_variable('b2', [self.config.label_size])
            y = tf.matmul(h, U) + b2
            if self.config.l2:
                tf.add_to_collection('total_loss', 0.5*self.config.l2*tf.nn.l2_loss(U))

        output = tf.nn.dropout(y, self.dropout_placeholder)
        return output

    def add_loss_op(self, y):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y,
                self.labels_placeholder))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        window = self.add_embedding()
        y = self.add_model(window)
        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_prediction = tf.equal(
            tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, input_labels, 
            shuffle=True, verbose=True):
        orig_X, orig_y = input_data, input_labels
        dp = self.config.dropout
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        for step, (x, y) in enumerate(
                data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                              label_size=self.config.label_size, shuffle=shuffle)):
            feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
            loss, total_correct, _ = session.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\t{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\t')
            sys.stdout.flush()
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

    def predict(self, session, X, y=None):
        dp = 1
        losses = []
        results = []
        if np.any(y):
            data = data_iterator(X, y, batch_size=self.config.batch_size,
                                 label_size=self.config.label_size, shuffle=False)
        else:
            data = data_iterator(X, batch_size=self.config.batch_size,
                                 label_size=self.config.label_size, shuffle=False)
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
        return np.mean(losses), results


def xavier_weight_init():

    def _xavier_initializer(shape, **kwargs):
        m = shape[0]
        n = shape[1] if len(shape) > 1 else shape[0]
        bound = np.sqrt(6) / np.sqrt(m + n)
        out = tf.random_uniform(shape, minval=-bound, maxval=bound)
        return out
    return _xavier_initializer


def print_confusion(confusion, num_to_tag):
    total_guessed_tags = confusion.sum(axis=0)
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / flaot(total_true_tags[i])
        print 'tag: {} - P {:2.4f}'.format(tag, prec, recall)


def calculate_confusion(config, predicted_indices, y_indices):
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion


def save_predictions(predictions, filename):
    with open(filename, 'wb') as f:
        for prediction in predictions:
            f.write(str(prediction) + '\n')


def test_NER():
    config = Config()
    with tf.Graph().as_default():
        model = NERModel(config)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            for epoch in xrange(config.max_epochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                train_loss, train_acc = model.run_epoch(session, model.X_train, model.y_train)
                val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Validation loss: {}'.format(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists('./weights'):
                        os.makedirs('./weights')
                    saver.save(session, 'weights/ner.weights')
                if epoch - best_val_epoch > config.early_stoppint:
                    break
                confusion = calcualte_confusion(config, predictions, model.y_dev)
                print_confusion(confusion, model.num_to_tag)
                print 'Total time {}'.format(time.time() - start)

            saver.restore(session, './weights/ner.weights')
            print 'Test'
            print '=-=-='
            print 'Writing predictions to test.predicted'
            _, predictions = model.predict(session, model.X_test, model.y_test)
            save_predictions(predictions, 'test.predicted')

if __name__ == '__main__':
    test_NER()
