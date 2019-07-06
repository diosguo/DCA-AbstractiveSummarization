import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten
from tensorflow.python.keras.models import Sequential

model = Sequential()
model.add(Embedding(3800,32,input_length=380))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(32, return_sequences=False), merge_mode='concat'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
