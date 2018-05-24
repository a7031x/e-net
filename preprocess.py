import re
import spacy
import ujson as json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from config import *
from utils import *


nlp = spacy.blank("en")


def remove_invalid_chars(text):
    return text.replace("''", '" ').replace("``", '" ')


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def string_features(passage, word_counter, char_counter):
    passage = remove_invalid_chars(passage)
    tokens = word_tokenize(passage)
    chars = [list(token) for token in tokens]
    for token in tokens:
        word_counter[token] += 1
        for char in token:
            char_counter[char] += 1
    return passage, tokens, chars


def process_word_embedding(word_counter):
    embedding_dict = {}
    with open(word_emb_file, 'r', encoding='utf8') as file:
        for line in file:
            array = line.split()
            word = ''.join(array[:-word_emb_dim])
            if word in word_counter:
                vector = list(map(float, array[-word_emb_dim:]))
                assert(len(vector) == 300)
                embedding_dict[word] = vector
    words = sorted(embedding_dict.keys())
    w2i = {token:idx for idx, token in enumerate(words, 1)}
    w2i[OOV] = 0
    i2w = {k:v for v,k in w2i.items()}
    embedding_dict[OOV] = [0. for _ in range(word_emb_dim)]
    word_embeddings = [embedding_dict[i2w[i]] for i in range(len(embedding_dict))]
    return word_embeddings, w2i
    

def process_char_embedding(char_counter):
    c2i = {
        OOV: 0
    }
    for char in char_counter:
        c2i[char] = len(c2i)
    char_embeddings = [np.random.normal(scale=0.01, size=char_emb_dim) for _ in range(len(c2i))]
    return char_embeddings, c2i


def process_dataset(filename, word_counter, char_counter):
    examples = []
    with open(filename, 'r', encoding='utf8') as file:
        source = json.load(file)
        for article in tqdm(source['data']):
            for para in article['paragraphs']:
                context, context_tokens, context_chars = string_features(para['context'], word_counter, char_counter)
                spans = convert_idx(context, context_tokens)
                for qa in para['qas']:
                    _, question_tokens, question_chars = string_features(qa['question'], word_counter, char_counter)
                    answers = []
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        answer_span = answer_span[0], answer_span[-1]
                        if answer_span not in answers:
                            answers.append(answer_span)
                    example = {
                        'context_tokens': context_tokens,
                        'context_chars': context_chars,
                        'question_tokens': question_tokens,
                        'question_chars': question_chars,
                        'answers': answers
                    }
                    examples.append(example)
    return examples


def build_features(examples, tfr_file, w2i, c2i):
    def extend_map(map, token):
        extend = [token, token.lower(), token.capitalize(), token.upper()]
        for each in extend:
            if each in map:
                return map[each]
        return map[OOV]
    def bytes_feature(value, map):
        if map is not None:
            value = [extend_map(map, x) for x in value]
        bytes = np.array(value).tostring()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes))
    with tf.python_io.TFRecordWriter(tfr_file) as writer:
        for example in examples:
            record = tf.train.Example(features={
                'context_tokens': bytes_feature(example['context_tokens'], w2i),
                'context_chars': bytes_feature(example['context_chars'], c2i),
                'question_tokens': bytes_feature(example['question_tokens'], w2i),
                'question_chars': bytes_feature(example['question_chars'], c2i),
                'answers': bytes_feature(example['answers'], None)
            })
            writer.write(record.SerializeToString())


def save_json(filename, obj, message=None):
    if message is not None:
        with open(filename, "w") as file:
            json.dump(obj, file)


mkdir('./generate/squad')
word_counter = Counter()
char_counter = Counter()
print('extracting examples...')
train_examples = process_dataset(train_file, word_counter, char_counter)
dev_examples = process_dataset(dev_file, word_counter, char_counter)
dev_examples, test_examples = dev_examples[:len(dev_examples)//2], dev_examples[len(dev_examples)//2:]
print('creating embeddings...')
word_embeddings, w2i = process_word_embedding(word_counter)
char_embeddings, c2i = process_char_embedding(char_counter)
print('#word: {}, #char: {}'.format(len(word_embeddings), len(char_embeddings)))
print('saving files...')
build_features(train_examples, train_record_file, w2i, c2i)
build_features(dev_examples, dev_record_file, w2i, c2i)
build_features(test_examples, test_record_file, w2i, c2i)
save_json(word_embeddings_file, word_embeddings)
save_json(char_embeddings_file, char_embeddings)
save_json(w2i_file, w2i)
save_json(c2i_file, c2i)
print('done.')
