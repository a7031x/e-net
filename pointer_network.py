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
            inp, start = func.pointer(d_match, init_state * self.dropout_mask, hidden_dim, mask)
            d_inp = tf.nn.dropout(inp, keep_prob=self.keep_prob)
            _, state = self.gru(d_inp, init_state)
            tf.get_variable_scope().reuse_variables()
            _, end = func.pointer(d_match, state * self.dropout_mask, hidden_dim, mask)
            return start, end
