import tensorflow as tf
import config
import utils
import pickle
import random


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [row + [fill] * (mlen - len(row)) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values


class Feeder:
    def __init__(self):
        self.w2i = utils.load_json(config.w2i_file)
        self.c2i = utils.load_json(config.c2i_file)


    def token_to_id(self, token):
        extend = [token, token.lower(), token.capitalize(), token.upper()]
        for each in extend:
            if each in self.w2i:
                return self.w2i[each]
        return self.w2i[config.OOV]


    def char_to_id(self, char):
        return self.c2i[char] if char in self.c2i else self.c2i[config.OOV]


    def toid(self, tokens):
        token_ids = [self.token_to_id(token) for token in tokens]
        char_ids = [[self.char_to_id(char) for char in token] for token in tokens]
        return token_ids, char_ids


class TrainFeeder(Feeder):
    def __init__(self, filename):
        super(TrainFeeder, self).__init__()
        with open(filename, 'rb') as file:
            self.examples = [x for x in pickle.load(file) if len(x['passage_tokens']) <= 400 and len(x['question_tokens']) <= 50]


    def parse(self, example):
        passage_tokens = example['passage_tokens']
        question_tokens = example['question_tokens']
        answer_starts = example['answer_starts']
        answer_ends = example['answer_ends']
        passage_token_ids, passage_char_ids = self.toid(passage_tokens)
        question_token_ids, question_char_ids = self.toid(question_tokens)
        y0 = answer_starts[-1]
        y1 = answer_ends[-1]
        label = [1 if y0 <= i <= y1 else 0 for i in range(len(passage_tokens))]
        return passage_token_ids, passage_char_ids, question_token_ids, question_char_ids, label


    @property
    def size(self):
        return len(self.examples)


    def prepare(self, shuffle=True):
        if shuffle:
            random.shuffle(self.examples)
        self.cursor = 0


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size=config.batch_size):
        size = min(self.size - self.cursor, batch_size)
        examples = self.examples[self.cursor:self.cursor+size]
        records = [self.parse(example) for example in examples]
        passage_token_ids, passage_char_ids, question_token_ids, question_char_ids, label = zip(*records)
        self.cursor += size
        return align2d(passage_token_ids), align3d(passage_char_ids), align2d(question_token_ids), align3d(question_char_ids), align2d(label)


if __name__ == '__main__':
    feeder = TrainFeeder(config.train_record_file)
    feeder.prepare()
    batch = feeder.next()
    ptids, pcids, qtids, qcids, label = batch
    print(pcids)