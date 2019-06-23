import os, sys, inspect, csv
from pathlib import Path
import statistics
import math
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, save
from bokeh.models.ranges import Range1d
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Line as Line_glyph
import numpy

# from loaddatasets.loaddata import loadDataLoosely

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# # Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
sys.path.append('../')
sys.path.append('../utils/')
sys.path.append('../loaddatasets/')
sys.path.append('../datavisulaization/')

from utils.evaluationMethods import *
from datavisualization.visutilsQt import drawSkelQt
from loaddatasets.loaddata import LoadData


# Load data
pathLearningData = '../data/'

# Segment Names
dataLoad = []
# Subject List
subjectList = ["P1", "P2"]

# Scenario List
scenarioList = ["Loose", "Tight"]

# Movement Numbers
movementNrList = list((range(1, 20)))

# Flag to decide calculation of joint orientation error or segment error
executeBlock = 2
if executeBlock == 2:
    fileName = 'JointOrientationError.csv'
else:
    fileName = 'SegmentErrorData.csv'

# If file exits, clear the content before writing
if Path(fileName).is_file():
    fileToTruncate = open(fileName, "w")
    fileToTruncate.truncate()
    fileToTruncate.close()

# Fixed list of segment pairs to find joint orientation errors
segmentPairs = {'Head': 'T8', 'RightUpperArm': ['T8', 'RightForeArm'], 'LeftUpperArm': ['T8', 'LeftForeArm'], 'RightForeArm': 'RightHand', 'LeftForeArm': 'LeftHand', 'T8': 'Pelvis', 'Pelvis': ['RightUpperLeg', 'LeftUpperLeg'], 'RightUpperLeg': 'RightLowerLeg', 'LeftUpperLeg': 'LeftLowerLeg', 'RightLowerLeg': 'RightFoot', 'LeftLowerLeg': 'LeftFoot'}
segmentNameList = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']

# Write the relevent infomation to the file
with open(fileName, 'w') as outcsv:
    fieldnames = []
    if(executeBlock == 1):
        # Write into 'SegmentErrorData.csv'
        fieldnames = ["Subject", "Movement", "Segment", "TotalAngleAvgError", "TotalEulerAvgError", "MeanSquaredError", "Standard Deviation"]
    else:
        # Write into 'JointOrientationError.csv'
        fieldnames = ["Subject", "Movement", "Segment1", "Segment2", "TotalAngleAvgError", "TotalEulerAvgError", "MeanSquaredError", "Standard Deviation"]

    writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
    writer.writeheader()         # Write the header into the file

    for subject in subjectList:                         # Subjects - P1 and P2
        for movement in movementNrList:                 # MovementList - 1 - 19
            for scenario in scenarioList:               # ScenarioList - Loosely and tightly
                LabelData_List = [subject, pathLearningData, scenario, "{:03d}".format(movement)]
                data = LoadData(skelColor=[1, 0, 0, 0.5])
                data.readData(LabelData_List, ['a', 'c', 'i'])
                dataLoad.append(data)

            # Condition to decide calculation of joint orientation error or segment error, 1 = segment error, 2 = joint orientation error
            if executeBlock == 1:
                # Calculation of segment error
                # segmentNameList = list(dataLoad[0].cData.segNamesByValue.values())
                for segment in segmentNameList:
                    meanSquaredError = 0
                    standardDeviation = 0
                    errSeq, errSeqEuler, avgErr, avgErrEuler = computeErrSegs([[dataLoad[0].getSegIdxByName(segment)]], dataLoad[0], dataLoad[1])
                    # for values in errSeq[0]:
                    #     meanSquaredError += (values - avgErr)**2
                    # meanSquaredError = meanSquaredError / len(errSeq[0])
                    # print("Mean Squared : ", meanSquaredError)
                    meanSquaredError = statistics.variance(errSeq[0])
                    standardDeviation = statistics.stdev(errSeq[0])
                    # print("standard deviation : ", standardDeviation)
                    # print("Variance : ", statistics.variance(errSeq[0]))
                    # print("Mean Error: ", meanSquaredError)
                     # writer.writerow({subject, movement, segment, avgErr, avgErrEuler})
                    writer.writerow({'Subject': subject, 'Movement': movement, 'Segment': segment, 'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
            else:
                # Calculation of joint orientation error
                for pairs in segmentPairs:
                    if isinstance(segmentPairs[pairs], list):
                        # If a key has multiple values, Ex. 'RightUpperArm': ['T8', 'RightForeArm'],
                        for value in segmentPairs[pairs]:
                            meanSquaredError = 0
                            standardDeviation = 0
                            errSeq, errSeqEuler, avgErr, avgErrEuler = computeErrSegs([[dataLoad[0].getSegIdxByName(pairs)], [dataLoad[0].getSegIdxByName(value)]], dataLoad[0], dataLoad[1])
                            # for values in errSeq[0]:
                            #     meanSquaredError += (values - avgErr) ** 2
                            # meanSquaredError = meanSquaredError / len(errSeq[0])
                            meanSquaredError = statistics.variance(errSeq[0])
                            standardDeviation = statistics.stdev(errSeq[0])
                            # print(errSeq, errSeqEuler, avgErr, avgErrEuler)
                            writer.writerow({'Subject': subject, 'Movement': movement, 'Segment1': pairs, 'Segment2':value,
                                             'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
                    else:
                        # If key has single value, Ex. 'Head': 'T8',
                        meanSquaredError = 0
                        standardDeviation = 0
                        errSeq, errSeqEuler, avgErr, avgErrEuler = computeErrSegs(
                            [[dataLoad[0].getSegIdxByName(pairs)], [dataLoad[0].getSegIdxByName(segmentPairs[pairs])]], dataLoad[0],
                            dataLoad[1])
                        # for values in errSeq[0]:
                        #     meanSquaredError += (values - avgErr) ** 2
                        # meanSquaredError = meanSquaredError / len(errSeq[0])
                        meanSquaredError = statistics.variance(errSeq[0])
                        standardDeviation = statistics.stdev(errSeq[0])
                        writer.writerow({'Subject': subject, 'Movement': movement, 'Segment1': pairs, 'Segment2': segmentPairs[pairs],
                                         'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
                        # print(errSeq, errSeqEuler, avgErr, avgErrEuler)

            dataLoad = []

# counter = 1
# meanSquaredErrorArrayForPlot = []
# standardDeviationArrayForPlot = []
# segmentsArrayForPlot = []
# avgErrorArrayForPlot = []
# eulerAvgErrorArrayForPlot = []
# subjectName = ''
#
# readErrorData = pd.read_csv('SegmentErrorData.csv', skiprows=[0], header=None)
# for rowsOfReadErrorData in readErrorData.values:
#     # if rowsOfReadErrorData[0] == 'P1':
#     if rowsOfReadErrorData[1] != counter or len(readErrorData) - 1:
#         output_file("../Plots/Error Plots/" + subjectName + " - " + str(counter) + ".html")
#         p = figure(x_range=segmentsArrayForPlot, plot_width=800, plot_height=800)
#         p.vbar(x=segmentsArrayForPlot, top=meanSquaredErrorArrayForPlot, width=0.9, legend='Mean Squared Error')
#         p.y_range.start = 0
#         p.xaxis.major_label_orientation = math.pi / 2
#
#         p.line(x=segmentsArrayForPlot, y=standardDeviationArrayForPlot, legend="Standard deviation", color='red',
#                line_width=2)
#         # p.line(x=segmentsArrayForPlot, y=avgErrorArrayForPlot, legend="Average Error", color='green')
#         # p.line(x=segmentsArrayForPlot, y=eulerAvgErrorArrayForPlot, legend="Average Euler Error", color='yellow')
#         p.legend.location = "top_left"
#         t = Title()
#         t.text = rowsOfReadErrorData[0] + " - " + str(counter) + " : Standard deviation and Mean squared error for segments"
#         p.title = t
#         save(p)
#         meanSquaredErrorArrayForPlot = []
#         standardDeviationArrayForPlot = []
#         segmentsArrayForPlot = []
#         # avgErrorArrayForPlot = []
#         # eulerAvgErrorArrayForPlot = []
#         # # End if
#
#     segmentsArrayForPlot.append(rowsOfReadErrorData[2])
#     meanSquaredErrorArrayForPlot.append(rowsOfReadErrorData[5])
#     standardDeviationArrayForPlot.append(rowsOfReadErrorData[6])
#     # avgErrorArrayForPlot.append(rows[3])
#     # eulerAvgErrorArrayForPlot.append(rows[4])
#
#     counter = rowsOfReadErrorData[1]
#     subjectName = rowsOfReadErrorData[0]


# with open('SegmentErrorData.csv', 'r') as incsv:
#     # reader = csv.reader(incsv)
#     # next(reader, None)
#     line_count = 0
#     csv_reader = csv.reader(incsv, delimiter=',')
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1










# with open('JointOrientationError.csv', 'w') as outcsv:
#     fieldnames = ["Subject", "Movement", "Segment1", "Segment2", "TotalAngleAvgError", "TotalEulerAvgError"]
#     writer = csv.DictWriter(outcsv, fieldnames = fieldnames)
#     writer.writeheader()

#Loosely coupled
# subject = "P1"
# scenario = "Loose"
# MovementNr = "{:03d}".format(5) #Todo: use this format to read the file
# print(MovementNr)
# LabelData_List = [subjectList[0], pathLearningData, scenarioList[0], "{:03d}".format(movementNrList[6])]
# color [R, G, B, transperancy]
# dataLoosely = LoadData(skelColor=[1, 0, 0, 0.5])
# dataLoosely.readData(LabelData_List, ['a', 'c'])
# print(dataLoosely)
# segmentNames = list(dataLoosely.cData.segNamesByValue.values())
# print(len(segmentNames))

#Tightly coupled
# scenario = "Tight"
# MovementNr = "{:03d}".format(5)
# LabelData_List = [subjectList, pathLearningData, scenario, MovementNr]
# dataTightly = LoadData(skelColor=[0, 1, 0,0.5])
# dataTightly.readData(LabelData_List, ['a', 'c'])

# Online Error plots
# just visualization (not online error-plots)
#errSegs = []
# Global rotation error between 'T8' and 'Pelvis' as online error-plot
# errSegs = [[dataLoosely.getSegIdxByName('T8'), dataLoosely.getSegIdxByName('Pelvis') ]]
# errSegs = [[dataLoosely.getSegIdxByName('LeftFoot')]]
# print(errSegs)
# Relative rotation error between 'T8' and 'Pelvis' as online error-plot
#errSegs = [[dataLoosely.getSegIdxByName('T8')], [dataLoosely.getSegIdxByName('Pelvis')]]
# Note also multiple errorplots are possible (just extend the list...)

# # Executing one or more tasks
# whatToDo = ['']
#
#
# if ('vis' in whatToDo):
#     print('Visualize data...')
#     # print(errSegs[0][0])
#     list_Loader = [dataLoosely, dataTightly]
#     drawSkelQt(list_Loader, errSegs)
#
# if ('eval' in whatToDo):
#     print('Example error computation...')
#     plotErrSegs(dataTightly, errSegs, computeErrSegs(errSegs, dataLoosely, dataTightly))
#
# if ('dataArrayForLearning' in whatToDo):
#     print('Example write-out of numpy array of data -> for learning')
#     nparray = dataLoosely.getArrayExportOfSequence()
#     print(nparray)
#     print(nparray.shape)
#     # Read-in an array of learned sequence data
#     dataLoosely.setQuatStatesFromArray(nparray)
#     # Test
#     #list_Loader = [dataLoosely, dataTightly]
#     #drawSkelQt(list_Loader, errSegs)
