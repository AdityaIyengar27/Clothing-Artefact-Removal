import inspect
import math
import os
import statistics
import sys
import csv
from pathlib import Path
import pandas as pd
from bokeh import events
from bokeh.models import CustomJS
from bokeh.models.annotations import Title
from bokeh.palettes import inferno
from bokeh.plotting import figure, output_file, save

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

# from utils.evaluationMethods import *
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.utils.evaluationMethods import *
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.loaddatasets.loaddata import LoadData
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.datavisualization.visutilsQt import drawSkelQt
# Load data
pathLearningData = '../data/'

# Funtion to find the Name of joint between 2 segements
def determineJointPositionBetween2Segemnts(segment1, segment2):
    if segment1 == 'Head' and segment2 == 'T8':
        return 'Neck'
    elif segment1 == 'RightUpperArm' and segment2 == 'T8':
        return 'Glenohumeral Right Joint'
    elif segment1 == 'LeftUpperArm' and segment2 == 'T8':
        return 'Glenohumeral Left Joint'
    elif segment1 == 'RightUpperArm' and segment2 == 'RightForeArm':
        return 'Right Elbow'
    elif segment1 == 'LeftUpperArm' and segment2 == 'LeftForeArm':
        return 'Left Elbow'
    elif segment1 == 'RightForeArm' and segment2 == 'RightHand':
        return 'Right Wrist'
    elif segment1 == 'LeftForeArm' and segment2 == 'LeftHand':
        return 'Left Wrist'
    elif segment1 == 'T8' and segment2 == 'Pelvis':
        return 'T8 - Pelvis'
    elif segment1 == 'Pelvis' and segment2 == 'RightUpperLeg':
        return 'Right Hip'
    elif segment1 == 'Pelvis' and segment2 == 'LeftUpperLeg':
        return 'Left Hip'
    elif segment1 == 'RightUpperLeg' and segment2 == 'RightLowerLeg':
        return 'Right Knee'
    elif segment1 == 'LeftUpperLeg' and segment2 == 'LeftLowerLeg':
        return 'Left Knee'
    elif segment1 == 'RightLowerLeg' and segment2 == 'RightFoot':
        return 'Right Ankle'
    elif segment1 == 'LeftLowerLeg' and segment2 == 'LeftFoot':
        return 'Left Ankle'


# Function to write angle for Loosely and Tightly Data
def makeAngleFile(csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle):
    looselyDataAngle.insert(0, jointPosition)
    looselyDataAngle.insert(0, 'Loose Data')
    tightlyDataAngle.insert(0, jointPosition)
    tightlyDataAngle.insert(0, 'Tight Data')
    csvWriter.writerow(looselyDataAngle)
    csvWriter.writerow(tightlyDataAngle)

# def makeMRPFile()
def makeMRPFile(csvWriter, jointPosition, MRPLooseData, MRPTightData):
    MRPLooseData.insert(0, jointPosition)
    MRPLooseData.insert(0, 'Loose MRP Data')
    MRPTightData.insert(0, jointPosition)
    MRPTightData.insert(0, 'Tight Data')
    csvWriter.writerow(MRPLooseData)
    csvWriter.writerow(MRPTightData)

# Segment Names
dataLoad = []
# Subject List
subjectList = ["P1", "P2"]

# Scenario List
scenarioList = ["Loose", "Tight"]

# Movement Numbers
movementNrList = list((range(1, 20)))

# Flag to decide calculation of joint orientation error or segment error
executeBlock = 5
if executeBlock == 2:
    fileName = 'JointOrientationError.csv'
elif executeBlock == 1:
    fileName = 'SegmentErrorData.csv'
elif executeBlock == 3:
    fileName = 'AngleData.csv'
elif executeBlock == 4:
    fileName = 'AngleDataDownSampled.csv'
elif executeBlock == 5:
    fileName = 'MRPData.csv'
# If file exits, clear the content before writing
# if Path(fileName).is_file():
#     fileToTruncate = open(fileName, "w")
#     fileToTruncate.truncate()
#     fileToTruncate.close()

# Fixed list of segment and segment pairs to find joint orientation errors
segmentPairs = {'Head': 'T8', 'RightUpperArm': ['T8', 'RightForeArm'], 'LeftUpperArm': ['T8', 'LeftForeArm'], 'RightForeArm': 'RightHand', 'LeftForeArm': 'LeftHand', 'T8': 'Pelvis', 'Pelvis': ['RightUpperLeg', 'LeftUpperLeg'], 'RightUpperLeg': 'RightLowerLeg', 'LeftUpperLeg': 'LeftLowerLeg', 'RightLowerLeg': 'RightFoot', 'LeftLowerLeg': 'LeftFoot'}
segmentNameList = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
movementList = ['', 'N-Pose', 'Three Step Calibration Palm Front', 'Three Step Calibration N-Pose', 'X Sense Gait Calibration', 'Trunk Flexion Extension', 'Truck Rotation', 'Trunk Side Bending', 'Squats', 'Upper Arm Flexion Extension', 'Upper Arm Abduction Closed Sleeve', 'Upper Arm Abduction Open Sleeve', 'Upper Arm Rotation Open Sleeve', 'Shrugging Open Sleeve', 'Lover Arm Flexion Open Sleeve', 'Lower Arm Flexion Upper Arm Flexed 90 Degree Open Sleeve', 'Lower Arm Rotation Open Sleeve', 'Picking Up from Floor - Putting to Shelf Open Sleeve', 'Picking Up from Floor - Walking - Putting Down Open Sleeve', 'Screwing Light Bulb Open Sleeve']
# csvWriter = {}
# Write the relevent infomation to the file
with open(fileName, 'w') as outcsv:
    fieldnames = []
    if(executeBlock == 1):
        # Write into 'SegmentErrorData.csv'
        fieldnames = ["Subject", "Movement", "Segment", "TotalAngleAvgError", "TotalEulerAvgError", "MeanSquaredError", "Standard Deviation"]
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()  # Write the header into the file
    elif executeBlock == 2:
        # Write into 'JointOrientationError.csv'
        fieldnames = ["Subject", "Movement", "Segment1", "Segment2", "TotalAngleAvgError", "TotalEulerAvgError", "MeanSquaredError", "Standard Deviation"]
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()  # Write the header into the file
    elif executeBlock == 3 or executeBlock == 4 or executeBlock == 5:
        # Write into 'AngleData.csv'
        csvWriter = csv.writer(outcsv, dialect='excel')
        # print()

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

                    errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs([[dataLoad[0].getSegIdxByName(segment)]], dataLoad[0], dataLoad[1])
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
                    # writer.writerow({'Subject': subject, 'Movement': movement, 'Segment': segment, 'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
            elif executeBlock == 2 or executeBlock == 3 or executeBlock == 4 or executeBlock == 5:
                # Calculation of joint orientation error
                for pairs in segmentPairs:
                    if isinstance(segmentPairs[pairs], list):
                        # If a key has multiple values, Ex. 'RightUpperArm': ['T8', 'RightForeArm'],
                        for value in segmentPairs[pairs]:
                            meanSquaredError = 0
                            standardDeviation = 0
                            jointPosition = determineJointPositionBetween2Segemnts(pairs, value)
                            errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs([[dataLoad[0].getSegIdxByName(pairs), dataLoad[0].getSegIdxByName(value)]], dataLoad[0], dataLoad[1])
                            # for values in errSeq[0]:
                            #     meanSquaredError += (values - avgErr) ** 2
                            # meanSquaredError = meanSquaredError / len(errSeq[0])
                            if executeBlock == 3 or executeBlock == 4:
                                makeAngleFile(csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle)
                            if executeBlock == 5:
                                makeMRPFile(csvWriter, jointPosition, MRPLooseData, MRPTightData)
                            meanSquaredError = statistics.variance(errSeq[0])
                            standardDeviation = statistics.stdev(errSeq[0])
                            # print(errSeq, errSeqEuler, avgErr, avgErrEuler)
                            # writer.writerow({'Subject': subject, 'Movement': movement, 'Segment1': pairs, 'Segment2':value,
                            #                  'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
                    else:
                        # If key has single value, Ex. 'Head': 'T8',
                        meanSquaredError = 0
                        standardDeviation = 0
                        jointPosition = determineJointPositionBetween2Segemnts(pairs, segmentPairs[pairs])
                        errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs(
                            [[dataLoad[0].getSegIdxByName(pairs), dataLoad[0].getSegIdxByName(segmentPairs[pairs])]], dataLoad[0],
                            dataLoad[1])

                        if executeBlock == 3 or executeBlock == 4:
                            makeAngleFile(csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle)
                        if executeBlock == 5:
                            makeMRPFile(csvWriter, jointPosition, MRPLooseData, MRPTightData)
                        # for values in errSeq[0]:
                        #     meanSquaredError += (values - avgErr) ** 2
                        # meanSquaredError = meanSquaredError / len(errSeq[0])
                        meanSquaredError = statistics.variance(errSeq[0])
                        standardDeviation = statistics.stdev(errSeq[0])
                        # writer.writerow({'Subject': subject, 'Movement': movement, 'Segment1': pairs, 'Segment2': segmentPairs[pairs],
                        #                  'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
                        # print(errSeq, errSeqEuler, avgErr, avgErrEuler)
            # drawSkelQt(dataLoad, [[dataLoad[0].getSegIdxByName(segment)]])
            dataLoad = []

# # todo : Uncomment the block below to create plots
# counter = 1                                       # Counter to control when to create the plot using bokeh
# meanSquaredErrorArrayForPlot = []                 # Array to hold all mean squared error for a particular movement
# standardDeviationArrayForPlot = []                # Array to hold all standard deviation for a particular movement
# segmentsArrayForPlot = []                         # Array to hold all segments for a particular movement
# avgErrorArrayForPlot = []                         # Array to hold all average for a particular movement - Not used currently
# eulerAvgErrorArrayForPlot = []                    # Array to hold all average Euler error for a particular movement - Not used currently
# subjectName = 'P1'
# path = ''
#
# if executeBlock == 1:
#     readErrorData = pd.read_csv('SegmentErrorData.csv', skiprows=[0], header=None)              # Read csv into a data frame for further processing
#     path = '../Plots/Error Plots/'                                                              # Define path to write graphs
# else:
#     readErrorData = pd.read_csv('JointOrientationError.csv', skiprows=[0], header=None)
#     path = '../Plots/Joint Orientation Error Plots/'
#
# # lastArrayValue = len(readErrorData) - 1
#
# for index, rowsOfReadErrorData in readErrorData.iterrows():
#     # if rowsOfReadErrorData[0] == 'P1':                                                      # Not required as we will process all data
#     if rowsOfReadErrorData[1] != counter or index == len(readErrorData) - 1:                           # Plot for the previous movement if the current movement changed. This means that all the data for the previous movement is plotted. But if it is the last movement then plot that, as we will not come in this block of code again.
#         print(subjectName)
#         output_file(path + subjectName + " - " + str(counter) + ".html")   # Name of the file to save the plot
#         p = figure(x_range=segmentsArrayForPlot, plot_width=800, plot_height=800)             # Create an initial figure
#         # p.vbar(x=segmentsArrayForPlot, top=meanSquaredErrorArrayForPlot, width=0.9, legend='Mean Squared Error')  # Plot bar graph for mean squared error
#         p.y_range.start = 0
#         p.xaxis.major_label_orientation = math.pi / 2
#
#         p.line(x=segmentsArrayForPlot, y=standardDeviationArrayForPlot, legend="Standard deviation", color='red',     # Plot line graph for Standard deviation
#                line_width=2)
#         p.line(x=segmentsArrayForPlot, y=avgErrorArrayForPlot, legend="Average Error", color='green')
#         # p.line(x=segmentsArrayForPlot, y=eulerAvgErrorArrayForPlot, legend="Average Euler Error", color='yellow')
#         p.legend.location = "top_left"
#         t = Title()
#         t.text = subjectName + " - " + movementList[counter] + " : Standard deviation and Mean squared error for segments"
#         p.title = t
#         save(p)                                                                               # Save the plot in a file
#         meanSquaredErrorArrayForPlot = []                                                     # Reset all arrays
#         standardDeviationArrayForPlot = []                                                    # Reset all arrays
#         segmentsArrayForPlot = []                                                             # Reset all arrays
#         avgErrorArrayForPlot = []
#         # eulerAvgErrorArrayForPlot = []
#         # # End if
#     if executeBlock == 1:
#         segmentsArrayForPlot.append(rowsOfReadErrorData[2])
#         meanSquaredErrorArrayForPlot.append(rowsOfReadErrorData[5])
#         standardDeviationArrayForPlot.append(rowsOfReadErrorData[6])
#         avgErrorArrayForPlot.append(rowsOfReadErrorData[3])
#     else:
#         # segmentPairs = {'Head': 'T8', 'RightUpperArm': ['T8', 'RightForeArm'], 'LeftUpperArm': ['T8', 'LeftForeArm'],
#         #                 'RightForeArm': 'RightHand', 'LeftForeArm': 'LeftHand', 'T8': 'Pelvis',
#         #                 'Pelvis': ['RightUpperLeg', 'LeftUpperLeg'], 'RightUpperLeg': 'RightLowerLeg',
#         #                 'LeftUpperLeg': 'LeftLowerLeg', 'RightLowerLeg': 'RightFoot', 'LeftLowerLeg': 'LeftFoot'}
#         if rowsOfReadErrorData[2] == 'Head' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Neck')
#         elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Glenohumeral Right Joint')
#         elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Glenohumeral Left Joint')
#         elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'RightForeArm':
#             segmentsArrayForPlot.append('Right Elbow')
#         elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'LeftForeArm':
#             segmentsArrayForPlot.append('Left Elbow')
#         elif rowsOfReadErrorData[2] == 'RightForeArm' and rowsOfReadErrorData[3] == 'RightHand':
#             segmentsArrayForPlot.append('Right Wrist')
#         elif rowsOfReadErrorData[2] == 'LeftForeArm' and rowsOfReadErrorData[3] == 'LeftHand':
#             segmentsArrayForPlot.append('Left Wrist')
#         elif rowsOfReadErrorData[2] == 'T8' and rowsOfReadErrorData[3] == 'Pelvis':
#             segmentsArrayForPlot.append('T8 - Pelvis')
#         elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'RightUpperLeg':
#             segmentsArrayForPlot.append('Right Hip')
#         elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'LeftUpperLeg':
#             segmentsArrayForPlot.append('Left Hip')
#         elif rowsOfReadErrorData[2] == 'RightUpperLeg' and rowsOfReadErrorData[3] == 'RightLowerLeg':
#             segmentsArrayForPlot.append('Right Knee')
#         elif rowsOfReadErrorData[2] == 'LeftUpperLeg' and rowsOfReadErrorData[3] == 'LeftLowerLeg':
#             segmentsArrayForPlot.append('Left Knee')
#         elif rowsOfReadErrorData[2] == 'RightLowerLeg' and rowsOfReadErrorData[3] == 'RightFoot':
#             segmentsArrayForPlot.append('Right Ankle')
#         elif rowsOfReadErrorData[2] == 'LeftLowerLeg' and rowsOfReadErrorData[3] == 'LeftFoot':
#             segmentsArrayForPlot.append('Left Ankle')
#
#         meanSquaredErrorArrayForPlot.append(rowsOfReadErrorData[6])
#         standardDeviationArrayForPlot.append(rowsOfReadErrorData[7])
#         avgErrorArrayForPlot.append(rowsOfReadErrorData[4])
#     # eulerAvgErrorArrayForPlot.append(rows[4])
#
#     counter = rowsOfReadErrorData[1]                                                          # Assign the current movement number every time to @var : counter, to keep track of change
#     subjectName = rowsOfReadErrorData[0]                                                      # Assign the subject name every time
#     # print(subjectName)
#
# # todo : Create plots for Joint Orientation errors
# counter = 1
# if executeBlock == 2:
#     output_file(
#         '../Plots/Anamalous PLots/' + "P1" + " - All Joint Errors" + ".html")  # Name of the file to save the plot
#     p = figure(x_range=['Neck', 'Glenohumeral Right Joint', 'Glenohumeral Left Joint', 'Right Elbow', 'Left Elbow', 'Right Wrist', 'Left Wrist', 'T8 - Pelvis', 'Right Hip', 'Left Hip', 'Right Knee', 'Left Knee', 'Right Ankle', 'Left Ankle'], plot_width=1400, plot_height=850)  # Create an initial figure
#     p.y_range.start = 0
#     myPalatte = inferno(19)
#
#     for index, rowsOfReadErrorData in readErrorData.iterrows():
#         if rowsOfReadErrorData[0] == 'P1' and rowsOfReadErrorData[1] != counter:                           # Plot for the previous movement if the current movement changed. This means that all the data for the previous movement is plotted. But if it is the last movement then plot that, as we will not come in this block of code again.
#             p.line(x=segmentsArrayForPlot, y=avgErrorArrayForPlot, legend=movementList[counter], color=myPalatte[counter], line_width=2)
#             avgErrorArrayForPlot = []
#             # # End if
#         if rowsOfReadErrorData[2] == 'Head' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Neck')
#         elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Glenohumeral Right Joint')
#         elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'T8':
#             segmentsArrayForPlot.append('Glenohumeral Left Joint')
#         elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'RightForeArm':
#             segmentsArrayForPlot.append('Right Elbow')
#         elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'LeftForeArm':
#             segmentsArrayForPlot.append('Left Elbow')
#         elif rowsOfReadErrorData[2] == 'RightForeArm' and rowsOfReadErrorData[3] == 'RightHand':
#             segmentsArrayForPlot.append('Right Wrist')
#         elif rowsOfReadErrorData[2] == 'LeftForeArm' and rowsOfReadErrorData[3] == 'LeftHand':
#             segmentsArrayForPlot.append('Left Wrist')
#         elif rowsOfReadErrorData[2] == 'T8' and rowsOfReadErrorData[3] == 'Pelvis':
#             segmentsArrayForPlot.append('T8 - Pelvis')
#         elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'RightUpperLeg':
#             segmentsArrayForPlot.append('Right Hip')
#         elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'LeftUpperLeg':
#             segmentsArrayForPlot.append('Left Hip')
#         elif rowsOfReadErrorData[2] == 'RightUpperLeg' and rowsOfReadErrorData[3] == 'RightLowerLeg':
#             segmentsArrayForPlot.append('Right Knee')
#         elif rowsOfReadErrorData[2] == 'LeftUpperLeg' and rowsOfReadErrorData[3] == 'LeftLowerLeg':
#             segmentsArrayForPlot.append('Left Knee')
#         elif rowsOfReadErrorData[2] == 'RightLowerLeg' and rowsOfReadErrorData[3] == 'RightFoot':
#             segmentsArrayForPlot.append('Right Ankle')
#         elif rowsOfReadErrorData[2] == 'LeftLowerLeg' and rowsOfReadErrorData[3] == 'LeftFoot':
#             segmentsArrayForPlot.append('Left Ankle')
#
#         avgErrorArrayForPlot.append(rowsOfReadErrorData[4])
#         counter = rowsOfReadErrorData[1]
#     # p.legend.location = "top_left"
#     p.xaxis.major_label_orientation = math.pi / 2
#     t = Title()
#     t.text = "P1" + " - " + "All Joint Errors" + " : Average angle error for segments"
#     p.title = t
#
#     def show_hide_legend(legend=p.legend[0]):
#         legend.visible = not legend.visible
#
#     p.js_on_event(events.DoubleTap, CustomJS.from_py_func(show_hide_legend))
#     p.legend.glyph_height = 5
#     p.legend.click_policy = "hide"
#     save(p)

# # Code to pre-process the data
# if executeBlock == 2:
#     with open(fileName, 'r') as incsv:
#         lines = list(incsv.read().split('\n'))[1:]
#
# movement = ''
# maxData = []
# jointTempList = []
# jointList = {}
# for line in lines:
#     mylist = line.split(',')
#     if mylist[0] == 'P1':
#         if mylist[1] != movement and movement != '':
#             maxValue = max(maxData)
#             for x in maxData:
#                 if float(x) > 0.5 * float(maxValue):
#                     jointTempList.append(maxData.index(x))
#             maxData = []
#             # for i in range(len(maxData):
#             jointList[movement] = jointTempList
#             jointTempList = []
#         maxData.append(mylist[4])
#         movement = mylist[1]

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
# whatToDo = ['vis']
#
# #
# if ('vis' in whatToDo):
#     print('Visualize data...')
    # print(errSegs[0][0])
    # list_Loader = [dataLoosely, dataTightly]
    # drawSkelQt(list_Loader, errSegs)
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
#     drawSkelQt(list_Loader, errSegs)
