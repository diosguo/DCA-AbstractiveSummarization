from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import tanh
import tensorflow as tf


class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.V = Dense(1)
        self.W1 = Dense(units)
        self.W2 = Dense(units)

    def call(self, query, value):
        hidden_state = tf.expand_dims(query,1)

        score = self.V(tf.nn.tanh(self.W1(query)+self.W2(value)))

        attention_weight = tf.nn.softmax(score,axis=1)

        context_vector = attention_weight * value

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weight

