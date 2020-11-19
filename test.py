import numpy as np
import argparse
import json
import io
import tensorflow as tf
import keras
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from model import create_model
from config import *

tokenizer_path = 'tokenizer.json'
tag_tokenizer_path = 'tag_tokenizer.json'

def parse_argument():
    parser = argparse.ArgumentParser(description='Bidirectional LSTM POS tagger')
    parser.add_argument('--sent', help='Enter your sentence')
    return parser.parse_args()

def load_tokenizer(path):
    with open(path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def get_tags(sequences, tag_index):
    sequence_tags = []
    for sequence in sequences:
        sequence_tag = []
        for categorical in sequence:
            sequence_tag.append(tag_index.get(np.argmax(categorical)))
        sequence_tags.append(sequence_tag)
    return sequence_tags
    
def predict():
    args = parse_argument()
    sentence = args.sent
    
    tokenizer = load_tokenizer(tokenizer_path)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    tag_tokenizer = load_tokenizer(tag_tokenizer_path)
    tag_index = tag_tokenizer.word_index
    tag_size = len(tag_index) + 1
    tokens = word_tokenize(sentence)
    print(f'tokens after being tokenized: {tokens}')
    print(f'tokens: {tokens}')
    encoded_sent = tokenizer.texts_to_sequences([tokens])[0]
    print(f'encoded sentence: {encoded_sent}')
    encoded_sent = pad_sequences([encoded_sent], maxlen=max_length, padding='post')
    print(f'encoded sentence after being padded: {encoded_sent}')
    model = create_model(vocab_size, max_length, embedding_dim, word_index, tag_index)
    model.load_weights('models/POS_BiLSTM_CRF_WSJ_new.h5')
    
    pred = model.predict(encoded_sent)
    sequence_tags = get_tags(pred, {i: t for t, i in tag_index.items()})
    print(sequence_tags[0][:len(tokens)])

if __name__ == "__main__":
    predict()
