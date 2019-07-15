#random numebr generation generation
import random
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

#assigning the max_length to the longest sequence found in the inputs
max_length= len(max(input_texts,key=len))

#partitioning the data to make training set
for makeTrain in range(len(input_texts)):
  train_range = random.randint(0,517)
  if((train_range not in hold_index) and (len(hold_index)<259)):
    hold_index.append(train_range)
    input_train_data.append(input_texts[train_range])
    target_train_data.append(target_texts[train_range])

max_train_input_length= len(max(input_train_data,key=len))
max_train_target_legth = len(max(target_train_data,key=len))

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



