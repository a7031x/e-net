import tensorflow as tf
import config
import utils
import numpy as np
from data import Feeder
from model import Model


def run_epoch(sess, model, feeder):
    feeder.prepare()
    while not feeder.eof():
        pw, pc, qw, qc, start, end = feeder.next()
        feed = model.feed(pw, pc, qw, qc, start, end, config.keep_prob)
        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed)
        print('loss: {:>.4F}'.format(loss))
        model.save(sess)


def train(auto_stop):
    word_embeddings = utils.load_json(config.word_embeddings_file)
    char_embeddings = utils.load_json(config.char_embeddings_file)
    model = Model(word_embeddings, char_embeddings, config.checkpoint_folder)
    feeder = Feeder(config.train_record_file)
    with tf.Session() as sess:
        model.restore(sess)
        run_epoch(sess, model, feeder)


train(False)