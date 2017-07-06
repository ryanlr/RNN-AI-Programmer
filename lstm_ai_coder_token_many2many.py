from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

path = "./jdk-tokens.txt"
filetext = open(path).read().lower()


slice = len(filetext)/100
slice = int (slice)
filetext = filetext[:slice]


tokenized = filetext.split()

print('# of tokens:', len(tokenized))

uniqueTokens = sorted(list(set(tokenized)))
print('total # of unique tokens:', len(uniqueTokens))
token_indices = dict((c, i) for i, c in enumerate(uniqueTokens))
indices_token = dict((i, c) for i, c in enumerate(uniqueTokens))

# cut the text in semi-redundant sequences of maxlen characters
NUM_INPUT_TOKENS = 10
step = 3
sequences = []


for i in range(0, len(tokenized) - NUM_INPUT_TOKENS-1, step):
    sequences.append(tokenized[i: i + NUM_INPUT_TOKENS+1])

print('# of training sequences:', len(sequences))

print('Vectorization...')
X_temp = np.zeros((len(sequences), NUM_INPUT_TOKENS + 1, len(uniqueTokens)), dtype=np.bool)
X = np.zeros((len(sequences), NUM_INPUT_TOKENS, len(uniqueTokens)), dtype=np.bool)
y = np.zeros((len(sequences), NUM_INPUT_TOKENS, len(uniqueTokens)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X_temp[i, t, token_indices[char]] = 1

num_sequences = len(X_temp)
for i, vec in enumerate(X_temp):
    y[i] = vec[1:]
    X[i]= vec[:-1]


# build the model: a single LSTM
print('Build model...')
model = Sequential()

# 1-layer LSTM
model.add(LSTM(128, input_shape=(NUM_INPUT_TOKENS, len(uniqueTokens)), return_sequences=True))

# 2-layer LSTM
#model.add(LSTM(128, return_sequences=True, input_shape=(NUM_INPUT_TOKENS, len(uniqueTokens))))
#model.add(LSTM(128, return_sequences=True))

model.add(TimeDistributed(Dense(len(uniqueTokens))))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print(model.summary())

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

    start_index = random.randint(0, len(tokenized) - NUM_INPUT_TOKENS - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = [] #''
        sequence = tokenized[start_index: start_index + NUM_INPUT_TOKENS]

        generated=list(sequence)

        print('----- Generating with seed: "' + ' '.join(sequence) + '"-------')
        sys.stdout.write(' '.join(generated))

        for i in range(100):
            x = np.zeros((1, NUM_INPUT_TOKENS, len(uniqueTokens)))
            for t, char in enumerate(sequence):
                x[0, t, token_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0][-1] # only get the last element
            next_index = sample(preds, diversity)
            next_pred_token = indices_token[next_index]

            sequence = sequence[1:]
            sequence.append(next_pred_token)

            sys.stdout.write(next_pred_token+" ")
            sys.stdout.flush()
        print()
