import numpy as np
import emoji
from emo_utils import *
import matplotlib.pyplot as plt

X_train, Y_train = read_csv('train_emoji.csv')
print(X_train)
X_test, Y_test = read_csv('tesss.csv')
maxLen = len(max(X_train, key=len).split())
index = 1
print(X_train[index], label_to_emoji(Y_train[index]))

Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)
index = 50
print(Y_train[index], "is converted into one hot", Y_oh_train[index])
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(r'E:\PycharmProjects\deeplearning-RNN-Week2-Operations on word vectors - Debiasing/glove.6B.50d.txt')

# word = 'cucumber'
# index = 289846
# print("the index of", word, "in the vocabulary is", word_to_index[word])
# print("the", str(index) + "th word in the vocabulary is", index_to_word[index])
#
#
# def sentence_to_avg(sentence, word_to_vec_map):
#     """
#     Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
#     and averages its value into a single vector encoding the meaning of the sentence.
#
#     Arguments:
#     sentence -- string, one training example from X
#     word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
#
#     Returns:
#     avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
#     """
#     words = sentence.lower().split()
#     avg = np.zeros(50)
#     for w in words:
#         avg += word_to_vec_map[w]
#     avg = avg / len(words)
#
#     return avg
#
# avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
# print("avg = ", avg)
#
#
# def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
#     """
#         Model to train word vector representations in numpy.
#
#         Arguments:
#         X -- input data, numpy array of sentences as strings, of shape (m, 1)
#         Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
#         word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
#         learning_rate -- learning_rate for the stochastic gradient descent algorithm
#         num_iterations -- number of iterations
#
#         Returns:
#         pred -- vector of predictions, numpy-array of shape (m, 1)
#         W -- weight matrix of the softmax layer, of shape (n_y, n_h)
#         b -- bias of the softmax layer, of shape (n_y,)
#     """
#
#     np.random.seed(1)
#
#     m = Y.shape[0]
#     n_y = 5
#     n_h = 50
#
#     W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
#     b = np.zeros((n_y, ))
#
#     Y_oh = convert_to_one_hot(Y, C=n_y)
#
#     for t in range(num_iterations):
#         for i in range(m):
#             avg = sentence_to_avg(X[i], word_to_vec_map)
#             z = np.dot(W, avg) + b
#             a = softmax(z)
#
#             cost = -sum(Y_oh[i] * np.log(a))
#
#             dz = a - Y_oh[i]
#             dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
#             db = dz
#
#             W = W - learning_rate * dW
#             b = b - learning_rate * db
#
#         if t % 100 == 0:
#             print("Epoch: " + str(t) + "---cost" + str(cost))
#
#     pred = predict(X, Y, W, b, word_to_vec_map)
#
#     return pred, W, b
#
# pred, W, b = model(X_train, Y_train, word_to_vec_map)
# print(pred)
#
# print("Training set:")
# pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
# print('Test set:')
# pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


def sentences_to_indices(X, word_to_index, max_len):
    """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]

    X_indices =  np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
      Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

      Arguments:
      word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
      word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

      Returns:
      embedding_layer -- pretrained layer Keras instance
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map['cucumber'].shape[0]

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)
    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
        Function creating the Emojify-v2 model's graph.

        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        model -- a model instance in Keras
    """

    sentences_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentences_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    model = Model(sentences_indices, X)

    return model

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)
model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# This code allows you to see the mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.
x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
