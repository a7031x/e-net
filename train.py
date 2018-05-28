import tensorflow as tf
import config
import utils
import numpy as np
from data import Feeder
from model import Model


def run_epoch(sess, model, feeder, writer):
    feeder.prepare()
    while not feeder.eof():
        pw, pc, qw, qc, start, end = feeder.next()
        feed = model.feed(pw, pc, qw, qc, start, end, config.keep_prob)
        summary, _, loss, logit_start, logit_end = sess.run([model.summary, model.optimizer, model.loss, model.logit_start, model.logit_end], feed_dict=feed)
        print('loss: {:>.4F}'.format(loss))
    #    print('logit_start[0]', max(logit_start[0]), np.argmax(logit_start[0]))
    #    print('logit_end[0]',max(logit_end[0]), np.argmax(logit_end[0]))
        model.save(sess)
        writer.add_summary(summary)


def train(auto_stop):
    word_embeddings = utils.load_json(config.word_embeddings_file)
    char_embeddings = utils.load_json(config.char_embeddings_file)
    model = Model(word_embeddings, char_embeddings, config.checkpoint_folder)
    feeder = Feeder(config.train_record_file)
    with tf.Session() as sess:
        model.restore(sess)
        utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize(writer)
        while True:
            run_epoch(sess, model, feeder, writer)


train(False)
