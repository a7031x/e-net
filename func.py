import tensorflow as tf

def softmax(value, mask):
    exp = tf.exp(value) * mask
    alpha = exp / tf.expand_dims(tf.reduce_sum(exp, -1), -1)
    return alpha


def dense(value, last_dim, use_bias=True, scope='dense'):
    with tf.variable_scope(scope):
        weight = tf.get_variable('weight', [value.get_shape()[-1], last_dim])
        out = tf.einsum('aij,jk->aik', value, weight)
        if use_bias:
            b = tf.get_variable('bias', [last_dim])
            out += b
        out = tf.identity(out, 'dense')
        return out


def attention_pooling(value, hidden_dim, mask):
    sj = tf.nn.tanh(dense(value, hidden_dim, scope='summary_sj'))
    uj = tf.squeeze(dense(sj, 1, use_bias=False, scope='summary_uj'), [-1])#[batch, len]
    alpha = softmax(uj, mask)#[batch, len]
    return tf.reduce_sum(tf.expand_dims(alpha, axis=-1) * value, axis=1), alpha


def summary(value, hidden_dim, mask, keep_prob, scope='summary'):
    with tf.variable_scope(scope):
        value = tf.nn.dropout(value, keep_prob=keep_prob)
        s, _ = attention_pooling(value, hidden_dim, mask)
        return s


def pointer(encoder_state, decoder_state, hidden_dim, mask, scope='pointer'):
    with tf.variable_scope(scope):
        length = tf.shape(encoder_state)[1]
        tiled_decoder_state = tf.tile(tf.expand_dims(decoder_state, axis=1), [1, length, 1])
        united_state = tf.concat([tiled_decoder_state, encoder_state], axis=2)
        _, alpha = attention_pooling(united_state, hidden_dim, mask)
        next_decoder_state = tf.reduce_sum(tf.expand_dims(alpha, axis=-1) * encoder_state, axis=1)
        return next_decoder_state, alpha


def cross_entropy(logit, target, mask):
    logit = tf.clip_by_value(logit, 1E-18, 1-1E-18)
    loss_t = -target * tf.log(logit) * mask
    loss_f = -(1-target) * tf.log(1-logit) * mask
    return loss_t + loss_f


def sparse_cross_entropy(logit, target, mask):
    one_hot = tf.one_hot(target, tf.shape(logit)[-1], dtype=tf.float32)
    loss = cross_entropy(logit, one_hot, mask)
    return tf.reduce_sum(loss, axis=-1)
