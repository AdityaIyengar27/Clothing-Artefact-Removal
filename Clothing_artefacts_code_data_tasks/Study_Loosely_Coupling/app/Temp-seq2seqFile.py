import numpy as np
import random
import keras, tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import backend as k
from keras import regularizers
inputData = []
targetData = []
inputCharacters = []
targetCharacters = []
input_train_data = []
target_train_data = []

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
k.tensorflow_backend._get_available_gpus()

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
    if(train_range not in hold_index) and (len(hold_index) <= len(inputData)/2):
        hold_index.append(train_range)
        input_train_data.append(inputData[train_range])
        target_train_data.append(targetData[train_range])


# max_train_input_length= len(max(input_train_data,key=len))
# max_train_target_legth = len(max(target_train_data,key=len))
length = 400
for value in input_train_data:
    for i in range(0, len(value), length):
        # tempInputList = np.zeros(length)
        tempInputList = value[i:i+length]
        tempInputListLength = len(tempInputList)
        while tempInputListLength < length:
            tempInputList.insert(tempInputListLength, 0)
            tempInputListLength += 1
            # tempInputList[len(tempInputList):length-1] = 0
            # tempInputList.append(0)
        inputCharacters.append(tempInputList)
        # print(np.array(inputCharacters).shape)
        # if innerValue not in inputCharacters:
        #     inputCharacters.add(innerValue)
    # tempInputList = []
inputCharacters = np.array(inputCharacters)
inputCharacters = inputCharacters.reshape(inputCharacters.shape[0], length, 1)
# print(inputCharacters.ndim)
# print(inputCharacters.size)
# inputCharacters = inputCharacters[:, :, 1]
print(inputCharacters.shape)
# input_train_data = array
for value in target_train_data:
    for i in range(0, len(value), length):
        tempOutputList = value[i:i+length]
        tempOutputListLength = len(tempOutputList)
        while tempOutputListLength < length:
            tempOutputList.insert(tempOutputListLength, 0)
            tempOutputListLength += 1
        targetCharacters.append(tempOutputList)

targetCharacters = np.array(targetCharacters)
targetCharacters = targetCharacters.reshape(targetCharacters.shape[0], length, 1)
# print(targetCharacters.ndim)
# print(targetCharacters.size)
# targetCharacters = targetCharacters[:, :, np.newaxis]
# targetCharacters = targetCharacters.reshape()
print(targetCharacters.shape)

# for i, (, target_train_data) in enumerate(zip(input_train_data, target_train_data)):
#     for t, char in enumerate(input_train_data):
#         encoder_input_data[i, t, input_token_index[char]] = 1
#     for t, char in enumerate(target_train_data):
#     #     decoder_target_data is ahead of decoder_input_data by one timestep
#         decoder_input_data[i, t, target_token_index[char]] = 1
#         if t > 0:
#         #decoder_target_data will be ahead by one timestep
#         # and will not include the start character.
#             decoder_target_data[i, t - 1, target_token_index[char]] = 1
#
# print(encoder_input_data[2].shape)
# print(decoder_target_data[2].shape)

batch_size = 64  # batch size for training
epochs = 100  # number of epochs to train for
latent_dim = 128  # latent dimensionality of

encoderInputs = Input(shape=(length, 1))
print(encoderInputs)
encoderLSTM = LSTM(latent_dim, return_state=True)
encoderOutputs, hiddenStates, CStates = encoderLSTM(encoderInputs)
encoderStates = [hiddenStates, CStates]
print(encoderStates)

decoderInputs = Input(shape=(length, 1))
print(decoderInputs)
decoderLSTM = LSTM(latent_dim, return_state=True, return_sequences=True)
decoderOutputs, _, _ = decoderLSTM(decoderInputs, initial_state=encoderStates)

model = Model(inputs=[encoderInputs, decoderInputs], outputs=decoderOutputs)
optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

model.fit([inputCharacters, targetCharacters], targetCharacters, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save('../Models/seq2seq.h5')

# decoderDense = Dense(len(tar))
        # if innerValue not in targetCharacters:
        #     targetCharacters.add(innerValue)
# inputCharacters = np.array(inputCharacters).reshape((len(inputCharacters), length, 1))
# targetCharacters = np.array(targetCharacters).reshape((len(targetCharacters), length, 1))
# print(np.array(inputCharacters).shape)
# print(np.array(targetCharacters).shape)
# print(len(inputCharacters[0]))

# max_encoder_seq_length = max([len(txt) for txt in input_train_data])
# max_decoder_seq_length = max([len(txt) for txt in target_train_data])
# num_encoder_tokens = len(inputCharacters)
# num_decoder_tokens = len(targetCharacters)
#
# print(num_encoder_tokens)
# print(num_decoder_tokens)
# input_token_index = dict(
#   [(char, i) for i, char in enumerate(inputCharacters)])
# target_token_index = dict(
#   [(char, i) for i, char in enumerate(targetCharacters)])
# print(len(input_token_index))

# encoder_input_data = np.zeros(
#   (len(input_train_data), max_encoder_seq_length, num_encoder_tokens),
#   dtype='float32')
# decoder_input_data = np.zeros(
#   (len(input_train_data), max_decoder_seq_length, num_decoder_tokens),
#   dtype='float32')
# decoder_target_data = np.zeros(
#   (len(input_train_data), max_decoder_seq_length, num_decoder_tokens),
#   dtype='float32')
# print(encoder_input_data.shape)
# print(decoder_input_data.shape)
# print(decoder_target_data.shape)
#
# for i, (input_train_data, target_train_data) in enumerate(zip(input_train_data, target_train_data)):
#     for t, char in enumerate(input_train_data):
#         encoder_input_data[i, t, input_token_index[char]] = 1
#     for t, char in enumerate(target_train_data):
#     #     decoder_target_data is ahead of decoder_input_data by one timestep
#         decoder_input_data[i, t, target_token_index[char]] = 1
#         if t > 0:
#         #decoder_target_data will be ahead by one timestep
#         # and will not include the start character.
#             decoder_target_data[i, t - 1, target_token_index[char]] = 1
#
# print(encoder_input_data[2].shape)
# print(decoder_target_data[2].shape)

# batch_size = 64  # batch size for training
# epochs = 100  # number of epochs to train for
# latent_dim = 256  # latent dimensionality of the encoding space

# encoder_inputs = Input(shape=(None, num_encoder_tokens))
# encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# encoder_states = [state_h, state_c]
#
# decoder_inputs = Input(shape=(None, num_decoder_tokens))
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state=encoder_states)
# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
#
# model = Model(inputs=[encoder_inputs, decoder_inputs],
#               outputs=decoder_outputs)
#
# model.compile(optimizer='Adam', kernel_regularizer=regularizers.l2(0.01))
# model.summary()
#
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)
# model.save('../Models/seq2seq.h5')
