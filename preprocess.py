import io
import json
import numpy as np
from itertools import groupby, islice
from build_vocabolary import build_vocab
from utils import get_word_tag, assign_unk
from collections import defaultdict
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from config import *

corpus_path = "WSJ_02-21.pos"
test_corpus_path = "WSJ_24.pos"

def get_raw_data(corpus_path):
    with open(corpus_path, 'r') as f:
        data = f.readlines()
    return data

def build_vocab2idx(corpus_path):
    vocab = build_vocab(corpus_path)
    vocab2idx = {}

    for i, tok in enumerate(sorted(vocab)):
        vocab2idx[tok] = i
    return vocab2idx

def get_data(data, vocab2idx):
    tokens = []
    tags []
    for toktag in training_data:
        tok, tag = get_word_tag(toktag, vocab2idx)
        tokens.append(tok.lower())
        tags.append(tag)
        
    sentences = [list(group) for k, group in groupby(tokens, lambda x: x == "--n--") if not k]
    labels = [list(group) for k, group in groupby(tags, lambda x: x == "--s--") if not k]
    return sentences, labels
    
def load_data():
    training_data = get_raw_data(corpus_path)
    vocab2idx = build_vocab2idx(corpus_path)

    # for training data
    sentences, labels = get_data(training_data, vocab2idx)
    
    train_index = int(len(sentences) * train_portion)
    x_train = sentences[:train_index]
    x_val = sentences[train_index:]
    y_train = labels[:train_index]
    y_val = labels[train_index:]

    tokenizer = Tokenizer(num_words=len(vocab2idx), oov_token='-OOV-')
    tokenizer.fit_on_texts(x_train)
    # save vocab
    tokenizer_json = tokenizer.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    max_length = len(max(x_train, key=len))
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    x_val = pad_sequences(x_val, maxlen=max_length, padding='post')

    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(y_train)
    tag_index = tag_tokenizer.word_index
    tag_size = len(tag_index) + 1 # for padding
    # save vocab for y
    tag_tokenizer_json = tag_tokenizer.to_json()
    with io.open('tag_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tag_tokenizer_json, ensure_ascii=False))

    y_train = tag_tokenizer.texts_to_sequences(y_train)
    y_val = tag_tokenizer.texts_to_sequences(y_val)

    y_train = pad_sequences(y_train, maxlen=max_length, padding='post')
    y_val = pad_sequences(y_val, maxlen=max_length, padding='post')

    y_train = [to_categorical(i, num_classes=tag_size) for i in y_train]
    y_val = [to_categorical(i, num_classes=tag_size) for i in y_val]
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # for testing data
    test_sentences, test_labels = get_data(testing_data, vocab2idx)
    x_test = tokenizer.texts_to_sequences(test_sentences)
    y_test = tag_tokenizer.texts_to_sequences(test_labels)

    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
    y_test = pad_sequences(y_test, maxlen=max_length, padding='post')

    y_test = [to_categorical(i, num_classes=tag_size) for i in y_test]
    y_test = np.asarray(y_test)

    
    
    return x_train, y_train, x_val, y_val, x_test, y_test, word_index, tag_index, max_length
    
    
    
