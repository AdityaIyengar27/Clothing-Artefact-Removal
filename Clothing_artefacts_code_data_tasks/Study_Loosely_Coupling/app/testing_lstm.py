from keras.models import load_model

import numpy as np
import random
import keras, tensorflow as tf
import itertools
import re
from ast import literal_eval
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, GRU
from keras import backend as k
from keras.regularizers import l2
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.utils.trafos import  MRP_to_quaternion
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.utils.evaluationMethods import Evaluation as eval

from keras import regularizers
inputData = []
targetData = []
inputCharacters = []
targetCharacters = []
input_test_data = []
target_test_data = []
targ = []
pred = []

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
k.tensorflow_backend._get_available_gpus()

#sanity testing as random number might generate the same number again and would be very bad for us!!
hold_index = []
hold_test_index = []
hold_val_index = []
predict_quat = []
target_quat = []
totalAngleErr = 0

with open('RelevantMRPDataWithSubject.csv', 'r') as f:
    lines = list(f.read().split('\n'))
    # data.append(lines)

for line in lines:
    if not line:
        continue
    mylist = line.split(',')
    if mylist[0] == 'P2':
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
            tempList = [i.strip("[]").strip("''").split(" ") for i in tempList]
            tempList = list(filter(None, itertools.chain(*tempList)))
            inputData.append(tempList)
            tempList = []
        else:
            # for value in mylist[2:]:
            #     if value not in targetCharacters:
            #         targetCharacters.add(value)
            # mylist.insert(2, '\t')
            # mylist.append('\n')
            tempList = mylist[3:]
            print(len(tempList),mylist[2])
            print(tempList)
            target_quat = tempList
            tempList = [i.strip("[]").strip("''").split(" ") for i in tempList]
            tempList = list(filter(None, itertools.chain(*tempList)))
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

length = 200
input_test_data = list(itertools.chain(*inputData))
noOfZeros = (length * 3) - (len(input_test_data) % (length * 3))
input_test_data = np.pad(input_test_data, (0, noOfZeros), 'constant')
inputCharacters = input_test_data.reshape(int(input_test_data.size / 600), length, 3)

target_test_data = list(itertools.chain(*targetData))

noOfZeros = (length * 3) - (len(target_test_data) % (length * 3))
target_test_data = np.pad(target_test_data, (0, noOfZeros), 'constant')
print(len(target_test_data))
targetCharacters = target_test_data.reshape(int(target_test_data.size / 600), length, 3)

print("input " , inputCharacters.shape)
print("target ", targetCharacters.shape)

model = load_model('../Models/seq2seqRelevantMRPWithLSTMRegularizertanh15_09_11.h5')

# print(model.metrics_names)
# evaluate = model.evaluate(inputCharacters,targetCharacters,verbose = 1)
predict = model.predict(inputCharacters)
print("predict ",predict.shape)
score,mean,acc = model.evaluate(targetCharacters,predict,batch_size = 64,verbose = 1)

#print("predict " , predict.shape)
#print("loss ", evaluate)
print('Test score:', score)
print('Test accuracy:', acc)
print('Test Error: ',  mean)
predict_data = list(itertools.chain(*predict))
#predict_data = list(itertools.chain(*predict_data))
predict_quat = predict_data[0:699]
#t = np.asarray(target_quat)
t = [float(i) for i in target_quat]
print(type(t))
t = np.asarray(t)
print(t.dtype)
a = MRP_to_quaternion(t)
b = MRP_to_quaternion(predict_quat)
c = eval.angleErrEuler(a,b)
print(c)

# for i in range(0,targetCharacters.shape[0]):
#     for j in range(0,length):
#         predict_quat.append(MRP_to_quaternion(predict[i][j]))
#
#            # print("predict ", predict[i][j])
#            # print("inside ",(targetCharacters[i][j]).astype(float))
#         target_quat.append(MRP_to_quaternion((targetCharacters[i][j]).astype(float)))
#         #print((predict[i][j]))
#     pred.append(predict_quat)
#     targ.append(target_quat)
#     predict_quat = []
#     target_quat = []
# print(len(pred))
# print(len(targ))
# flattened_list = [y for x in pred for y in x]
# print(len(flattened_list[0]))
# for k in range(0,len(pred)):
#     for m in range(0,len(pred[k])):
#     #a = 0
#         q2 = targ[k][m]
#    # print(float(pred[k]))
#    # print(float(targ[k]))
#     #a = pred[k] + targ[k]
#    # print (a)
#    #  print(len(pred[k]))
#    #  print(len(targ[k]))
#         totalAngleErr += eval.angleErrEuler(pred[k][m],targ[k][m])
#
# print(totalAngleErr)
#MRP_to_quaternion(mrp)

#batch_size=batch_size
#works well when the diffrence is huge

#input ['-4.13465095e-01' '3.85770091e-03' '-2.71683239e-04']
#target ['-0.41321205' '-0.01035728' '-0.01373655']

#predict [-0.33760396  0.05542773 -0.07545152]

# doesn't work so well if the difference is very low

# input ['-0.41402208' '0.00449149' '-0.00514275']
# target ['-0.41311927' '-0.00294469' '-0.01291335']
#
# predict [-0.31287804  0.07813162 -0.07069097]

#there is a possibility that if the prediction is good in one/two axes the third axis calculation is sacrificed

# input ['-1.16279438' '18.03883789' '-2.96063223']
# target ['-0.6117004' '21.41372703' '-0.36903183']
# predict [-0.60742414 21.619549    0.07451182]

# input ['-0.61330674' '18.7218817' '-3.57646145']
# target ['-0.55178407' '22.26794399' '-0.61384444']
# predict [-0.60744005 21.619501    0.07452064]


# input ['-0.46859367' '19.43223535' '-3.32010823']
# target ['-0.37243172' '23.24435612' '-0.53464336']
#
# predict [-0.5759373  21.61255     0.08944773]

#lstms results with different way of data partitioning
#input  ['-0.02162612' '0.00334506' '-0.00836384']
#target  ['-0.01231863' '0.00569533' '0.00255037']
#predict  [-0.00925488 -0.01461554  0.00848632]
#
# input  ['-0.00513915' '0.04088172' '-0.02453846']
# target  ['0.00855263' '0.02320397' '0.01521706']
# predict  [-0.01145926 -0.00876798  0.00716196]

#input  ['-0.00513915' '0.04088172' '-0.02453846']
#target  ['0.00855263' '0.02320397' '0.01521706']
#predict  [-0.01145926 -0.00876798  0.00716196]


# input  ['-0.00513915' '0.04088172' '-0.02453846']
# target  ['0.00855263' '0.02320397' '0.01521706']
#predict  [-0.00841414 -0.00303677 -0.00585897]


# input  ['-0.00344795' '-0.03506994' '-0.00161784']
# target  ['-0.00904686' '-0.02030162' '-0.00668837']
# predict  [-0.00184235 -0.03163702  0.01230465]
# predict  [-0.01161132 -0.00349514 -0.00506604]


# input  ['-0.00344795' '-0.03506994' '-0.00161784']
# target  ['-0.00904686' '-0.02030162' '-0.00668837']
# predict  [-0.01161132 -0.00349514 -0.00506604]

#input  ['-0.41110317' '0.06353068' '-0.17101965']
# target  ['-0.42056253' '0.03795554' '-0.27931182']
#predict  [-0.07686238 -0.02262503  0.05620463]