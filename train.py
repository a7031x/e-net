import tensorflow as tf
import config
import utils
import numpy as np
from data import TrainFeeder
from model import Model


def run_epoch(sess, model, feeder, writer):
    feeder.prepare()
    while not feeder.eof():
        pw, pc, qw, qc, label = feeder.next()
        feed = model.feed(pw, pc, qw, qc, label, config.keep_prob)
        summary, _, loss, global_step = sess.run([model.summary, model.optimizer, model.loss, model.global_step], feed_dict=feed)
        model.save(sess)
        writer.add_summary(summary, global_step=global_step)
        print('loss: {:>.4F}'.format(loss))


def train(auto_stop):
    word_embeddings = utils.load_json(config.word_embeddings_file)
    char_embeddings = utils.load_json(config.char_embeddings_file)
    model = Model(word_embeddings, char_embeddings, config.checkpoint_folder)
    feeder = TrainFeeder(config.train_record_file)
    with tf.Session() as sess:
        model.restore(sess)
        #utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize(writer)
        while True:
            run_epoch(sess, model, feeder, writer)


train(False)
