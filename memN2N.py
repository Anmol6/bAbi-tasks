import numpy as np
import tensorflow as tf


def temp_encoding(buffer_size, embedding_size):
    T = tf.Variable(
        tf.random_normal([buffer_size, embedding_size], stddev=0.2))
    return T


def pos_encoding(sentence_length, embedding_size):
	position_encoding = np.ones(
	    (sentence_length, embedding_size), dtype=np.float32)


    for j in range(sentence_length+1):
        for k in range(embedding_size+1):
        	position_encoding[j][k] = (1-(j/sentence_length)) - (k/embedding_size)*(1-2*(j/sentence_length))
    return position_encoding

class memN2N(object):

    def __init__(nhops = 3, Q, Ans, S, embedding_size=25, vocab_size, buffer_size, sentence_length, temporal_encoding=true):
        self._nhops = nhops
        self._Q = Q
        self._Ans = Ans
        self._S = S
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._temporal_encoding = temporal_encoding
        self._buffer_size = buffer_size
        self._sentence_length = sentence_length
        self._t_init = tf.random_normal_initializer(mean = 0.0,stddev=0.2)

        build_model()

    def build_model():
        self.A = tf.Variable(
            self._t_init([self._vocab_size, self._embedding_size]))
        self.C = tf.Variable(
            self._t_init([self._vocab_size, self._embedding_size]))
        self.B = tf.Variable(
            self._t_init([self._vocab_size, self._embedding_size]))

        self._stories = tf.placeholder(
            tf.int32, shape=(None, self._buffer_size, self._sentence_length))
        self._queries = tf.placeholder(
            tf.int32, shape=(None, self._sentence_length))

        self._answer = tf.placeholder(tf.int32, shape=(None, vocab_size))

        if (self._temporal_encoding):
            self.T_A = temp_encoding(self._buffer_size, self._embedding_size)
            self.T_C = temp_encoding(self._buffer_size, self._embedding_size)

        self._pos_encoding = pos_encoding(self._sentence_length,self._embedding_size)
    def inference():
        m0 = tf.nn.embedding_lookup(self.A, self._stories)
        q0 = tf.nn.embedding_lookup(self.B, self._queries)

        m = tf.reduce_sum(m0*self._pos_encoding, 2) + self.T_A
        q_emb = tf.reduce_sum(q0*self._pos_encoding,1)

      





