
train_file = './data/squad/train-v1.1.json'
dev_file = './data/squad/dev-v1.1.json'
#test_file = './data/squad/dev-v1.1.json'
word_emb_file = './data/glove/glove.840B.300d.txt'
train_record_file = './generate/squad/train.tfrecords'
dev_record_file = './generate/squad/dev.tfrecords'
test_record_file = './generate/squad/test.tfrecords'
word_embeddings_file = './generate/word_embeddings.json'
char_embeddings_file = './generate/char_embeddings.json'
w2i_file = './generate/w2i.json'
c2i_file = './generate/c2i.json'

word_emb_dim = 300
char_emb_dim = 8
OOV = '--OOV--'