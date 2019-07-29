import numpy as np
import random
import keras, tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
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
# print(RepeatVector(2))
batch_size = 64  # batch size for training
epochs = 100  # number of epochs to train for
latent_dim = 128  # latent dimensionality of
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(length, 1)))
print(model.layers[0].input_shape)
print(model.layers[0].output_shape)
model.add(RepeatVector(length))
model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(1)))
# model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(length, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model = Model(inputs=[encoderInputs, decoderInputs], outputs=decoderOutputs)
optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()
# fit model
model.fit(inputCharacters, targetCharacters, verbose=0, batch_size=1, epochs=epochs, validation_split=0.2)
model.save('../Models/seq2seq.h5')