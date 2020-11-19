
def create_model(vocab_size, max_length, embedding_dim, word_index):
    embeddings_index = {}
    with open('/tmp/glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.strip().split()
            curr_word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[curr_word] = coefs
    embeddings_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i > vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                    input_length=max_length, weights=[embeddings_matrix], mask_zero=True))
    model.add(Bidirectional(LSTM(units=embedding_dim, return_sequences=True, 
                             recurrent_dropout=0.01)))
    model.add(TimeDistributed(Dense(len(tag_index))))
    # model.add(Activation('relu'))
    crf = CRF(len(tag_index), sparse_target=True)
    model.add(crf)
    model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])
    mode.summary()
    return model
