import numpy as np
import random
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense

inputData = []
targetData = []
inputCharacters = set()
targetCharacters = set()
input_train_data = []
target_train_data = []


#sanity testing as random number might generate the same number again and would be very bad for us!!
hold_index = []
hold_test_index = []
hold_val_index = []

with open('AngleDataDownSampled.csv', 'r') as f:
    lines = list(f.read().split('\n'))
    # data.append(lines)

for line in lines:
    mylist = line.split(',')
    if mylist[0] == 'Loose Data':
        # for value in mylist[2:]:
        #     if value not in inputCharacters:
        #         inputCharacters.add(value)
        # mylist.append('\n')
        inputData.append(mylist[2:])

    else:
        # for value in mylist[2:]:
        #     if value not in targetCharacters:
        #         targetCharacters.add(value)
        # mylist.insert(2, '\t')
        # mylist.append('\n')
        targetData.append(mylist[2:])

#partitioning the data to make training set
for makeTrain in range(len(inputData)):
  train_range = random.randint(0, 517)
  if((train_range not in hold_index) and (len(hold_index)<259)):
    hold_index.append(train_range)
    input_train_data.append(inputData[train_range])
    target_train_data.append(targetData[train_range])


# max_train_input_length= len(max(input_train_data,key=len))
# max_train_target_legth = len(max(target_train_data,key=len))
for value in input_train_data:
    for innerValue in value:
        if innerValue not in inputCharacters:
            inputCharacters.add(innerValue)

for value in target_train_data:
    for innerValue in value:
        if innerValue not in targetCharacters:
            targetCharacters.add(innerValue)

max_encoder_seq_length = max([len(txt) for txt in input_train_data])
max_decoder_seq_length = max([len(txt) for txt in target_train_data])
num_encoder_tokens = len(inputCharacters)
num_decoder_tokens = len(targetCharacters)

print(num_encoder_tokens)
print(num_decoder_tokens)
input_token_index = dict(
  [(char, i) for i, char in enumerate(inputCharacters)])
target_token_index = dict(
  [(char, i) for i, char in enumerate(targetCharacters)])
# print(len(input_token_index))

encoder_input_data = np.zeros(
  (len(input_train_data), max_encoder_seq_length, num_encoder_tokens),
  dtype='float32')
decoder_input_data = np.zeros(
  (len(input_train_data), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
decoder_target_data = np.zeros(
  (len(input_train_data), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_target_data.shape)

for i, (input_train_data, target_train_data) in enumerate(zip(input_train_data, target_train_data)):
    for t, char in enumerate(input_train_data):
        encoder_input_data[i, t, input_token_index[char]] = 1
    for t, char in enumerate(target_train_data):
    #     decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1
        if t > 0:
        #decoder_target_data will be ahead by one timestep
        # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1

print(encoder_input_data[2].shape)
print(decoder_target_data[2].shape)

batch_size = 64  # batch size for training
epochs = 100  # number of epochs to train for
latent_dim = 256  # latent dimensionality of the encoding space

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs],
              outputs=decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('/results/seq2seq_eng-ger.h5')
