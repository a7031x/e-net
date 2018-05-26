import tensorflow as tf
import func

class PointerNetwork:
    def __init__(self, batch, hidden, keep_prob=1.0, scope="pointer_network"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.scope = scope
        self.keep_prob = keep_prob
        self.dropout_mask = tf.nn.dropout(tf.ones([batch, hidden], dtype=tf.float32), keep_prob=keep_prob)


    def __call__(self, init_state, match, hidden_dim, mask):
        with tf.variable_scope(self.scope):
            d_match = tf.nn.dropout(match, keep_prob=self.keep_prob)
            decoder_state, start = func.pointer(d_match, init_state * self.dropout_mask, hidden_dim, mask)
            decoder_state = tf.nn.dropout(decoder_state, keep_prob=self.keep_prob)
            _, next_state = self.gru(decoder_state, init_state)
            tf.get_variable_scope().reuse_variables()
            _, end = func.pointer(d_match, next_state * self.dropout_mask, hidden_dim, mask)
            return start, end
