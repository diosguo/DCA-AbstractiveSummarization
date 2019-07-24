from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Softmax
from tensorflow.keras.activations import tanh
import keras.backend as K
import tensorflow as tf

class WordAttention(Model):
    def __init__(self, units):
        super(WordAttention, self).__init__()
        self.W5 = Dense(units)
        self.W6 = Dense(units)
        self.V = Dense(1)
        self.softmax = Softmax(axis=-1)

    def call(self, query, value):
        query = K.expand_dims(query, 1)
        l_a = []
        c = []
        for agent_index, agent_encode in enumerate(value):
            l = self.softmax( self.V(tanh(self.W5(agent_encode)+self.W6(query))))
            c.append(Lambda(lambda x: tf.reduce_sum(x, axis=1))(l*agent_encode))

        return K.stack(c, axis=1)


class AgentAttention(Model):
    def __init__(self, units):
        super(AgentAttention, self).__init__()
        self.V = Dense(1)
        self.W7 = Dense(units)
        self.W8 = Dense(units)
        self.softmax = Softmax(axis=-1)

    def call(self, query, value):
        query = K.expand_dims(query,1)
        g = self.softmax(self.V(tanh(self.W7(value)+self.W8(query))))
        c = Lambda(lambda x:tf.reduce_sum(x, axis=1))(g*value)
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
        query = tf.expand_dims(query,1)

        score = self.V(tf.nn.tanh(self.W1(query)+self.W2(value)))

        attention_weight = tf.nn.softmax(score,axis=1)

        context_vector = attention_weight * value

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weight
