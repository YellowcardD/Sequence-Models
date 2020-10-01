from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

K.clear_session()
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# print("X.shape:", X.shape)
# print("Y.shape:", Y.shape)
# print("Xoh.shape:", Xoh.shape)
# print("Yoh.shape:", Yoh.shape)

# index = 0
# print("Source date:", dataset[index][0])
# print("Target date:", dataset[index][1])
# print()
# print("Source after preprocessing (indices):", X[index])
# print("Target after preprocessing (indices):", Y[index])
# print()
# print("Source after preprocessing (one-hot):", Xoh[index])
# print("Target after preprocessing (one-hot):", Yoh[index])

# define shared layers as global variales
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation='tanh')
densor2 = Dense(1, activation='relu')
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
    """

    s_prev = repeator(s_prev) # shape of (m, Tx, 2*n_a)
    concat = concatenator([a, s_prev]) # shape of (m, Tx, 2*n_a+n_s)
    e = densor1(concat) # shape of (m, Tx, 10)
    energies = densor2(e) # shape of (m, Tx, 1)
    alphas = activator(energies) # shape of (m, Tx, 1)
    context = dotor([alphas, a]) # shape of (m, 1, 2*n_a)

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
    """

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s, ), name='s0')
    c0 = Input(shape=(n_s, ), name='c0')
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    for t in range(Ty):

        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(initial_state=[s, c], inputs=context)
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model

print(human_vocab['<pad>'])
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

#model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
model.load_weights('model.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    print(source.shape)
    source = source.reshape(1, source.shape[0], source.shape[1])
    print(source.shape)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))