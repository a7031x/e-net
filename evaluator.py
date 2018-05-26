import numpy as np

def run_epoch(sess, model, feeder):
    feeder.prepare()
    while not feeder.eof():
        pw, pc, qw, qc, start, end = feeder.next()
        feed = model.feed(pw, pc, qw, qc, start, end, 1.0)
        start, end = sess.run([model.output_start, model.output_end], feed_dict=feed)


class Evaluator:
    def evaluate(self, sess, model, feeder):
        feeder.prepare(False)
        correct = 0
        total = 0
        while not feeder.eof():
            feed = model_feed(model, feeder, withdict)
            batch_size, logits, labels = sess.run([model.batch_size, model.logits, model.tag_input], feed_dict=feed)
            for sent, tags in zip(logits, labels):
                for char, tag in zip(sent, tags):
                    if tag != -1:
                        if np.argmax(char) == tag:
                            correct += 1
                        total += 1
            total += batch_size
        return correct / total