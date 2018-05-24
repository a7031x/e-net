import tensorflow as tf
import config

class Model:
    def __init__(self, word_embeddings, char_embeddings):
        initializer = tf.random_uniform_initializer(-0.5, 0.5)
        with tf.variable_scope('model', initializer=initializer):
            self.initialize()


    def initialize(self):
        self.create_inputs()
        self.create_embeddings(self.num_chars, self.embedding_dim)
        self.create_lstm()
        self.create_logits()
        self.create_loss()
        self.create_optimizers()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_folder)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('MODEL LOADED.')
        else:
            sess.run(tf.global_variables_initializer())


    def save(self, sess):
        self.saver.save(sess, os.path.join(self.ckpt_folder, 'model.ckpt'))