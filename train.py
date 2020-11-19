import numpy as np
from preprocess import load_data
from config import *
from model import create_model
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
import crf
from crf import CRF

x_train, y_train, x_val, y_val, x_test, y_test, word_index, tag_index, maxlen = load_data()
max_length = maxlen
vocab_size = len(word_index) + 1

model = create_model(vocab_size, max_length, embedding_dim, word_index):
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data(x_val, y_val), verbose=1)
model.save_weights('models/1stmodel.h5')

viterbi_acc = model.evaluate(x_test, y_test)
print(f"{viterbi_acc[1] * 100}")
