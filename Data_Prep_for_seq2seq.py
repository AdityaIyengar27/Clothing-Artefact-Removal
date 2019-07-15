#random numebr generation generation
import random
import statistics
import numpy as np
#to generate one hot encodings for floating point numbers
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
# enc = OrdinalEncoder()

import numpy as np
import tensorflow as tf
import helpers


#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model

#
# def unique(list1):
#   # intilize a null list
#   unique_list = []
#
#   # traverse for all elements
#   for x in list1:
#     # check if exists in unique_list or not
#     if x not in unique_list:
#       unique_list.append(x)
#       # print list
#   return unique_list


with open("AngleData.csv", 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')
data = []
for i in range(len(lines)):
  data.append(list(lines[i].split(",")))



#print((data[3]))
#print(len(data))


#sanity testing as random number might generate the same number again and would be very bad for us!!
hold_index = []
hold_test_index = []
hold_val_index = []
#input_texts has the data from scenario :Loose
input_texts = []
input_train_data = []
input_val_data = []
input_test_data = []
#target_texts has the data from scenario :Tight
target_texts = []
target_train_data = []
target_val_data = []
target_test_data = []
# maximum length for the unique integer encoding vector
max_length = 0



#
# print(data[1][2:])
for j in range(len(data)):
  if(data[j][0] == "Loose Data"):
    input_texts.append(data[j][2:])
  elif(data[j][0] == "Tight Data"):
    target_texts.append(data[j][2:])

# for u in range(len(input_texts)):
#
#   print(len(unique(input_texts[u])))
#   print(len(input_texts[u]))
#assigning the max_length to the longest sequence found in the inputs
max_value = []
min_value = []
total_values=[]
mean_value = []

max_length= len(max(input_texts,key=len))
for c in range(len(input_texts)):
  max_value.append(max(input_texts[c]))
  min_value.append(min(input_texts[c]))
  total_values.append(len(input_texts[c]))
  mean_value.append(statistics.mean(list(map(float, input_texts[c]))))
print(max_value)
print(min_value)
print(total_values)
print(mean_value)


#partitioning the data to make training set
for makeTrain in range(len(input_texts)):
  train_range = random.randint(0,517)
  if((train_range not in hold_index) and (len(hold_index)<259)):
    hold_index.append(train_range)
    input_train_data.append(input_texts[train_range])
    target_train_data.append(target_texts[train_range])

max_train_input_length= len(max(input_train_data,key=len))
max_train_target_legth = len(max(target_train_data,key=len))
min_train_input_length= len(min(input_train_data,key=len))
min_train_taget_length= len(min(target_train_data,key=len))
print(max_train_input_length)

#partitioning the data to make test set
for makeTest in range(len(input_texts)):
  test_range = random.randint(0,517)
  if((test_range not in hold_index) and (test_range not in hold_test_index) and  (len(hold_test_index)<163)):
    hold_test_index.append(test_range)
    input_test_data.append(input_texts[test_range])
    target_test_data.append(target_texts[test_range])

# print(len(input_test_data))
# print(len(target_test_data))

#partitioning the data to make validation set
for makeVal in range(len(input_texts)):
  val_range = random.randint(0,517)
  if((val_range not in hold_test_index) and (val_range not in hold_test_index) and (val_range not in hold_val_index) and (len(hold_val_index)<96)):
    hold_val_index.append(val_range)
    input_val_data.append(input_texts[val_range])
    target_val_data.append(target_texts[val_range])

# print(len(input_val_data))
# print(len(target_val_data))
#
# #Model design begins
input_token_index = []
target_token_index = []
for dataLen in range(len(input_texts)):
    input_token_index.append(dict(
    [(u,char) for u, char in enumerate(input_texts[dataLen])]))

    target_token_index.append(dict(
    [(u,char) for u, char in enumerate(target_texts[dataLen])]))

# print((input_token_index[0]))
# print((target_token_index[0]))
# num_encoder_tokens = 0
# num_decoder_tokens = 0
# for u in range(int(len((input_train_data)))):
#   num_encoder_tokens += len(input_train_data[u])
# print (num_encoder_tokens)

# for f in range(int(len((target_train_data)))):
#   num_decoder_tokens += len(target_train_data[f])
# print (num_decoder_tokens)

encoder_input_data = np.zeros(
  (len(input_train_data), max_train_input_length, min_train_input_length),
  dtype='float32')
decoder_input_data = np.zeros(
  (len(input_train_data), max_train_target_legth, min_train_taget_length),
  dtype='float32')
decoder_target_data = np.zeros(
  (len(input_train_data), max_train_target_legth, min_train_taget_length),
  dtype='float32')


#
# for line in lines:
#     mylist = line.split(',')
#     if mylist[0] == 'Loose Data':
#         for value in mylist[2:]:
#             if value not in inputCharacters:
#                 inputCharacters.add(value)
#         # mylist.append('\n')
#         inputData.append(mylist[2:])
#
#     else:
#         for value in mylist[2:]:
#             if value not in targetCharacters:
#                 targetCharacters.add(value)
#         # mylist.insert(2, '\t')
#         # mylist.append('\n')
#         targetData.append(mylist[2:])


for i, (input_text, target_text) in enumerate(zip(input_train_data, target_train_data)):

  for t, char in enumerate(input_text):
    print(t,char)
    encoder_input_data[i, t, input_token_index[t]] = 1.
  for t, char in enumerate(target_text):
    # decoder_target_data is ahead of decoder_input_data by one timestep
    decoder_input_data[i, t, target_token_index[t]] = 1.
    if t > 0:
      # decoder_target_data will be ahead by one timestep
      # and will not include the start character.
      decoder_target_data[i, t - 1, target_token_index[t]] = 1.