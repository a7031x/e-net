import re
import spacy
import ujson as json
import numpy as np
import tensorflow as tf
import config
import pickle
from tqdm import tqdm
from collections import Counter
from utils import mkdir, save_json

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
    with open(config.word_emb_file, 'r', encoding='utf8') as file:
        for line in file:
            array = line.split()
            word = ''.join(array[:-config.word_emb_dim])
            if word in word_counter:
                vector = list(map(float, array[-config.word_emb_dim:]))
                assert(len(vector) == 300)
                embedding_dict[word] = vector
    words = sorted(embedding_dict.keys())
    w2i = {token:idx for idx, token in enumerate(words, 2)}
    w2i[config.NULL] = 0
    w2i[config.OOV] = 1
    i2w = {k:v for v,k in w2i.items()}
    embedding_dict[config.NULL] = [0. for _ in range(config.word_emb_dim)]
    embedding_dict[config.OOV] = [0. for _ in range(config.word_emb_dim)]
    word_embeddings = [embedding_dict[i2w[i]] for i in range(len(embedding_dict))]
    return word_embeddings, w2i
    

def process_char_embedding(char_counter):
    c2i = {
        config.NULL: 0,
        config.OOV: 1
    }
    for char in char_counter:
        c2i[char] = len(c2i)
    char_embeddings = [np.random.normal(scale=0.01, size=config.char_emb_dim) for _ in range(len(c2i))]
    return char_embeddings, c2i


def process_dataset(filename, word_counter, char_counter):
    examples = []
    with open(filename, 'r', encoding='utf8') as file:
        source = json.load(file)
        for article in tqdm(source['data']):
            for para in article['paragraphs']:
                context, context_tokens, _ = string_features(para['context'], word_counter, char_counter)
                spans = convert_idx(context, context_tokens)
                for qa in para['qas']:
                    _, question_tokens, _ = string_features(qa['question'], word_counter, char_counter)
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
                        'passage_tokens': context_tokens,
                        'question_tokens': question_tokens,
                        'answer_starts': [s for s,_ in answers],
                        'answer_ends': [s for _,s in answers]
                    }
                    examples.append(example)
    return examples


def build_features(examples, filename):
    with open(filename, 'wb') as file:
        pickle.dump(examples, file)


mkdir('./generate/squad')
word_counter = Counter()
char_counter = Counter()
print('extracting examples...')
train_examples = process_dataset(config.train_file, word_counter, char_counter)
dev_examples = process_dataset(config.dev_file, word_counter, char_counter)
dev_examples, test_examples = dev_examples[:len(dev_examples)//2], dev_examples[len(dev_examples)//2:]
print('creating embeddings...')
word_embeddings, w2i = process_word_embedding(word_counter)
char_embeddings, c2i = process_char_embedding(char_counter)
print('#word: {}, #char: {}'.format(len(word_embeddings), len(char_embeddings)))
print('saving files...')
build_features(train_examples, config.train_record_file)
build_features(dev_examples, config.dev_record_file)
build_features(test_examples, config.test_record_file)
save_json(config.word_embeddings_file, word_embeddings)
save_json(config.char_embeddings_file, char_embeddings)
save_json(config.w2i_file, w2i)
save_json(config.c2i_file, c2i)
print('done.')
