import numpy as np
import preprocess
import utils
import config
from data import Feeder, align2d

def run_epoch(sess, model, feeder):
    feeder.prepare()
    while not feeder.eof():
        pw, pc, qw, qc, start, end = feeder.next()
        feed = model.feed(pw, pc, qw, qc, start, end, 1.0)
        start, end = sess.run([model.output_start, model.output_end], feed_dict=feed)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class Evaluator(Feeder):
    def __init__(self):
        super(Evaluator, self).__init__()


    def create_feed(self, passage, question):
        _, passage_tokens, _ = preprocess.string_features(passage)
        _, question_tokens, _ = preprocess.string_features(question)
        passage_tids, passage_cids = self.toid(passage_tokens)
        question_tids, question_cids = self.toid(question_tokens)
        return passage_tids, align2d(passage_cids), question_tids, align2d(question_cids), passage_tokens


    def evaluate(self, sess, model, passage, question):
        pt, pc, qt, qc, words = self.create_feed(passage, question)
        feed = model.feed([pt], [pc], [qt], [qc])
        logit = sess.run(model.logit, feed_dict=feed)[0]
        ids = [id for id in range(len(logit)) if logit[id] > 0]
        selected = [words[id] for id in ids]
        wl = sorted(list(zip(words, logit)), key=lambda x:x[1], reverse=True)
        print(wl[:5])
        


if __name__ == '__main__':
    from model import Model
    import tensorflow as tf
    word_embeddings = utils.load_json(config.word_embeddings_file)
    char_embeddings = utils.load_json(config.char_embeddings_file)
    model = Model(word_embeddings, char_embeddings, config.checkpoint_folder)
    evaluator = Evaluator()
    with tf.Session() as sess:
        model.restore(sess)
        #evaluator.evaluate(sess, model, 'The cat sat on the mat', 'what is on the mat')
        evaluator.evaluate(sess, model, "The debating chamber of the Scottish Parliament has seating arranged in a hemicycle, which reflects the desire to encourage consensus amongst elected members. There are 131 seats in the debating chamber. Of the total 131 seats, 129 are occupied by the Parliament's elected MSPs and 2 are seats for the Scottish Law Officers – the Lord Advocate and the Solicitor General for Scotland, who are not elected members of the Parliament but are members of the Scottish Government. As such the Law Officers may attend and speak in the plenary meetings of the Parliament but, as they are not elected MSPs, cannot vote. Members are able to sit anywhere in the debating chamber, but typically sit in their party groupings. The First Minister, Scottish cabinet ministers and Law officers sit in the front row, in the middle section of the chamber. The largest party in the Parliament sits in the middle of the semicircle, with opposing parties on either side. The Presiding Officer, parliamentary clerks and officials sit opposite members at the front of the debating chamber.",
        'What is the seating arrangement of the debating chamber?')
        evaluator.evaluate(sess, model, "The Annual Conference, roughly the equivalent of a diocese in the Anglican Communion and the Roman Catholic Church or a synod in some Lutheran denominations such as the Evangelical Lutheran Church in America, is the basic unit of organization within the UMC. The term Annual Conference is often used to refer to the geographical area it covers as well as the frequency of meeting. Clergy are members of their Annual Conference rather than of any local congregation, and are appointed to a local church or other charge annually by the conference's resident Bishop at the meeting of the Annual Conference. In many ways, the United Methodist Church operates in a connectional organization of the Annual Conferences, and actions taken by one conference are not binding upon another.",
        "The term Annual Conference is often used to refer to what?")