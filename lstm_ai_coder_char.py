from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

path = "./jdk-chars.txt"
text = open(path).read()
slice = len(text)/5
slice = int(slice)

# slice the text to make training faster
text = text[:slice]

print('# of characters in file:', len(text))

chars = sorted(list(set(text)))
print('# of unique chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

NUM_INPUT_CHARS = 40
STEP = 3
sequences = []
next_chars = []

for i in range(0, len(text) - NUM_INPUT_CHARS, STEP):
    sequences.append(text[i: i + NUM_INPUT_CHARS])
    next_chars.append(text[i + NUM_INPUT_CHARS])

print('# of training samples:', len(sequences))

print('Vectorize training data')
X = np.zeros((len(sequences), NUM_INPUT_CHARS, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single layer LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(NUM_INPUT_CHARS, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, epochs=1)

    start_index = random.randint(0, len(text) - NUM_INPUT_CHARS - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sequence = text[start_index: start_index + NUM_INPUT_CHARS]
        generated += sequence
        print('----- Generating with seed: "' + sequence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, NUM_INPUT_CHARS, len(chars)))
            for t, char in enumerate(sequence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sequence = sequence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
