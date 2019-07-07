import keras
from keras.layers import Dense, LSTM, Bidirectional, Flatten, Embedding
from keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Embedding(1,10,input_length=1))
model.add(LSTM(10,return_sequences=True))
model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.summary()
LSTM
print(model.layers[1].states)