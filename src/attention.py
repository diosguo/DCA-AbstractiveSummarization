from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Softmax
from tensorflow.python.keras.activations import tanh
import tensorflow as tf


class WordAttention(Model):
    def __init__(self, units):
        super(WordAttention, self).__init__()
        self.W5 = Dense(units)
        self.W6 = Dense(units)
        self.V = Dense(1)
        self.softmax = Softmax(axis=-1)

    def call(self, query, value):
        query = tf.expand_dims(query, 1)
        l_a = []
        c = []
        for agent_index, agent_encode in enumerate(value):
            l = self.softmax( self.V(tanh(self.W5(agent_encode)+self.W6(query))))
            c.append(tf.reduce_sum(l*agent_encode, axis=1))
        return tf.stack(c, axis=1)


class AgentAttention(Model):
    def __init__(self, units):
        super(AgentAttention, self).__init__()
        self.V = Dense(1)
        self.W7 = Dense(units)
        self.W8 = Dense(units)
        self.softmax = Softmax(axis=-1)

    def call(self, query, value):
        query = tf.expand_dims(query,1)
        g = self.softmax(self.V(tanh(self.W7(value)+self.W8(query))))
        c = tf.reduce_sum(g*value, axis=1)
        return c


class HiAttention(Model):
    def __init__(self, units):
        super(HiAttention, self).__init__()
        self.word = WordAttention(units)
        self.agent = AgentAttention(units)

    def call(self, query, value):
        c_a = self.word(query,value)
        context_vector = self.agent(query, c_a)

        return context_vector



class BahAttention(Model):
    def __init__(self, units):
        super(BahAttention, self).__init__()
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
