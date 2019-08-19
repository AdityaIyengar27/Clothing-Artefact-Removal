import numpy as np
import random
import keras, tensorflow as tf
from statistics import mean
import itertools
import re
from ast import literal_eval
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, GRU
from keras import backend as k
from keras.regularizers import l2

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

with open('RelevantMRPDataWithSubject.csv', 'r') as f:
    lines = list(f.read().split('\n'))
    # data.append(lines)

length = 200

for line in lines:
    if not line:
        continue
    mylist = line.split(',')
    if mylist[0] == 'P1':
        if mylist[1] == 'Loose MRP Data':
            # for value in mylist[2:]:
            #     print()
            #     if value not in inputCharacters:
            #         inputCharacters.add(value)
            # mylist.append('\n')
            # tempList = mylist[2:]
            # for x in range(0, len(tempList)):
                # tempList.append(literal_eval(x))
            tempList = mylist[3:]
            # print(len(tempList))
            # print(tempList)
            tempList = [i.strip("[]").strip("''").split(" ") for i in tempList]
            tempList = list(filter(None, itertools.chain(*tempList)))
            tempList = [float(i) for i in tempList]
            paddingValues = mean(tempList)
            noOfPaddingValues = (length * 3) - (len(tempList) % (length * 3))
            tempList = np.pad(tempList, (0, noOfPaddingValues), 'constant', constant_values=paddingValues)
            inputData.append(tempList)
            tempList = []
        else:
            # for value in mylist[2:]:
            #     if value not in targetCharacters:
            #         targetCharacters.add(value)
            # mylist.insert(2, '\t')
            # mylist.append('\n')
            tempList = mylist[3:]
            # print(len(tempList))
            # print(tempList)
            tempList = [i.strip("[]").strip("''").split(" ") for i in tempList]
            tempList = list(filter(None, itertools.chain(*tempList)))
            tempList = [float(i) for i in tempList]
            paddingValues = mean(tempList)
            noOfPaddingValues = (length * 3) - (len(tempList) % (length * 3))
            tempList = np.pad(tempList, (0, noOfPaddingValues), 'constant', constant_values=paddingValues)
            targetData.append(tempList)
            tempList = []

# print(inputData[0])
# partitioning the data to make training set
# for makeTrain in range(len(inputData)):
#     if makeTrain[0] == 'P1' and makeTrain[1] == 'Loose':
#         input_train_data.append(makeTrain[5:])
#     elif makeTrain[0] == 'P1' and makeTrain[1] == 'Tight':
#         target_train_data.append(makeTrain[5:])
#     print(makeTrain)
    # train_range = random.randint(0, 517)
    # if(train_range not in hold_index) and (len(hold_index) <= len(inputData)/2):
    #     hold_index.append(train_range)
    #     input_train_data.append(inputData[train_range])
    #     target_train_data.append(targetData[train_range])
# print(inputData[0])
# print(targetData[0])

# print(inputData[0])
input_train_data = np.asarray(list(itertools.chain(*inputData)))
print(input_train_data.shape)
# noOfZeros = (length * 3) - (len(input_train_data) % (length * 3))
# input_train_data = np.pad(input_train_data, (0, noOfZeros), 'constant')
inputCharacters = input_train_data.reshape(int(input_train_data.size / 600), length, 3)

target_train_data = np.asarray(list(itertools.chain(*targetData)))
# noOfZeros = (length * 3) - (len(target_train_data) % (length * 3))
# target_train_data = np.pad(target_train_data, (0, noOfZeros), 'constant')
targetCharacters = target_train_data.reshape(int(target_train_data.size / 600), length, 3)

# print(inputCharacters.shape)
# print(targetCharacters.shape)

batch_size = 64  # batch size for training
epochs = 200  # number of epochs to train for
# latent_dim = 256
latent_dim = inputCharacters.shape[1]  # latent dimensionality of

# define model
# activationfn = keras.layers.LeakyReLU(alpha=0.3)
model = Sequential()
model.add(LSTM(latent_dim, activation='tanh', input_shape=(length, 3), kernel_regularizer=l2(0.0001)))
print(model.layers[0].input_shape)
print(model.layers[0].output_shape)
model.add(RepeatVector(latent_dim))
model.add(LSTM(latent_dim, activation='tanh'))
model.add(RepeatVector(latent_dim))
model.add(LSTM(latent_dim, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(3)))
# model.add(Dense(1))
print(model.layers[1].input_shape)
print(model.layers[1].output_shape)
print(model.layers[2].input_shape)
print(model.layers[2].output_shape)

# model.add(TimeDistributed(Dense(1)))
# model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(length, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model = Model(inputs=[encoderInputs, decoderInputs], outputs=decoderOutputs)
optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'acc'])
model.summary()
# fit model
historyObject = model.fit(inputCharacters, targetCharacters, verbose=2, batch_size=batch_size, epochs=epochs)
print(historyObject.history)
model.save('../Models/seq2seqRelevantMRPWithLSTMRegularizertanh19_09_19.h5')
