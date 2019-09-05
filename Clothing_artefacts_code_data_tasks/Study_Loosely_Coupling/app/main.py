import inspect
import math
import os
import statistics
import sys
import csv
from pathlib import Path
import pandas as pd

# from loaddatasets.loaddata import loadDataLoosely
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.app.createplots import createplotsforsegments, \
    createplotsforjointOrientation

executeBlock = 0

# Function to ask to pass arguments
def printArguments ():
    print("Please enter one of the following number as an argument")
    print("1. Calculate Segment Error")
    print("2. Joint Orientation Error")
    print("3. Calculate Angle Data")
    print("4. Down-sample Angle Data")
    print("5. Calculate MRP Data")
    print("6. Calculate Relevant MRP Data. This requires Joint Orientation Error to be calculated already.")

# To check if the number of arguments is at least 2
if len(sys.argv) != 2:
    executeBlock = 0
elif len(sys.argv) == 2:
    filename, executeBlock = sys.argv

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# Use this if you want to include modules from a subfolder
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
def makeAngleFile(subject, csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle):
    looselyDataAngle.insert(0, jointPosition)
    looselyDataAngle.insert(0, 'Loose Data')
    looselyDataAngle.insert(0, subject)
    tightlyDataAngle.insert(0, jointPosition)
    tightlyDataAngle.insert(0, 'Tight Data')
    tightlyDataAngle.insert(0, subject)
    csvWriter.writerow(looselyDataAngle)
    csvWriter.writerow(tightlyDataAngle)

# Function to write MRP data
def makeMRPFile(subject, csvWriter, movement, jointPosition, MRPLooseData, MRPTightData):
    # flag = conditionalDataReduction()
    MRPLooseData.insert(0, jointPosition)
    MRPLooseData.insert(0, 'Loose MRP Data')
    MRPLooseData.insert(0, movement)
    MRPLooseData.insert(0, subject)
    MRPTightData.insert(0, jointPosition)
    MRPTightData.insert(0, 'Tight MRP Data')
    MRPTightData.insert(0, movement)
    MRPTightData.insert(0, subject)
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

# Fixed list of segment, segment pairs, movement list and joint mapping to find joint orientation errors
segmentPairs = {'Head': 'T8', 'RightUpperArm': ['T8', 'RightForeArm'], 'LeftUpperArm': ['T8', 'LeftForeArm'], 'RightForeArm': 'RightHand', 'LeftForeArm': 'LeftHand', 'T8': 'Pelvis', 'Pelvis': ['RightUpperLeg', 'LeftUpperLeg'], 'RightUpperLeg': 'RightLowerLeg', 'LeftUpperLeg': 'LeftLowerLeg', 'RightLowerLeg': 'RightFoot', 'LeftLowerLeg': 'LeftFoot'}
segmentNameList = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
movementList = ['', 'N-Pose', 'Three Step Calibration Palm Front', 'Three Step Calibration N-Pose', 'X Sense Gait Calibration', 'Trunk Flexion Extension', 'Truck Rotation', 'Trunk Side Bending', 'Squats', 'Upper Arm Flexion Extension', 'Upper Arm Abduction Closed Sleeve', 'Upper Arm Abduction Open Sleeve', 'Upper Arm Rotation Open Sleeve', 'Shrugging Open Sleeve', 'Lover Arm Flexion Open Sleeve', 'Lower Arm Flexion Upper Arm Flexed 90 Degree Open Sleeve', 'Lower Arm Rotation Open Sleeve', 'Picking Up from Floor - Putting to Shelf Open Sleeve', 'Picking Up from Floor - Walking - Putting Down Open Sleeve', 'Screwing Light Bulb Open Sleeve']
jointListMapping = {0: 'Neck', 1: 'Glenohumeral Right Joint', 3: 'Glenohumeral Left Joint', 2: 'Right Elbow', 4: 'Left Elbow', 5: 'Right Wrist', 6: 'Left Wrist', 7: 'T8 - Pelvis', 8: 'Right Hip', 9: 'Left Hip',10: 'Right Knee',11: 'Left Knee',12: 'Right Ankle',13: 'Left Ankle'}

# Function to set the threshold of the joint orientation angle error
def thresholdjointOrientationAngle(data):
    maxValue = max(data)
    for x in data:
        if float(x) > 0.5 * float(maxValue):
            # joint = determineJointPositionBetween2Segemnts(mylist[2], mylist[3])
            jointTempList.append(jointListMapping.get(data.index(x)))
    return jointTempList

jointListP1 = {}
jointListP2 = {}

# Flag to decide calculation
if executeBlock == 2:
    fileName = 'JointOrientationError.csv'
elif executeBlock == 0:
    printArguments()
elif executeBlock == 1:
    fileName = 'SegmentErrorData.csv'
elif executeBlock == 3:
    fileName = 'AngleData.csv'
elif executeBlock == 4:
    fileName = 'AngleDataDownSampled.csv'
elif executeBlock == 5:
    fileName = 'MRPDataWithSubject.csv'
elif executeBlock == 6:
    # Code to pre-process the data ie to use only those joint orientations whose error value is greater than 50% of the maximum value for that movement
    # ------------------------------------------------------------------------------------------------------ #
    fileName = 'RelevantMRPDataWithSubject.csv'
    with open('JointOrientationError.csv', 'r') as incsv:
        lines = list(incsv.read().split('\n'))[1:]
    movement = ''
    maxData = []
    jointTempList = []
    mylist = []
    for line in lines:
        if (lines.index(line) != len(lines) - 1):
            mylist = line.split(',')
            if mylist == '' or (mylist[1] != movement and movement != ''):
                # Find the threshold error value and eliminate those joints whose value is less than 50% of the max value for the movement
                jointTempList = thresholdjointOrientationAngle(maxData)
                # for i in range(len(maxData):
                if(person == 'P1'):
                    jointListP1[movement] = jointTempList
                elif(person == 'P2'):
                    jointListP2[movement] = jointTempList
                jointTempList = []
                maxData = []
            maxData.append(mylist[4])
            person = mylist[0]
            movement = mylist[1]
        else:
            jointTempList = thresholdjointOrientationAngle(maxData)
            # for i in range(len(maxData):
            if (person == 'P1'):
                jointListP1[movement] = jointTempList
            elif (person == 'P2'):
                jointListP2[movement] = jointTempList
            jointTempList = []
            maxData = []
    # ------------------------------------------------------------------------------------------------------ #

# If file exits, clear the content before writing
if Path(fileName).is_file():
    fileToTruncate = open(fileName, "w")
    fileToTruncate.truncate()
    fileToTruncate.close()

# Create a csv writer
csvWriter = {}

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
    elif executeBlock == 3 or executeBlock == 4 or executeBlock == 5 or executeBlock == 6:
        # Write into 'AngleData.csv' or 'AngleDataDownSampled.csv' or 'MRPDataWithSubject.csv' or 'RelevantMRPDataWithSubject.csv'
        csvWriter = csv.writer(outcsv, dialect='excel')
        # print()

    for subject in subjectList:                         # Subjects - P1 and P2
        for movement in movementNrList:                 # MovementList - 1 - 19
            for scenario in scenarioList:               # ScenarioList - Loosely and tightly
                # Load the data
                LabelData_List = [subject, pathLearningData, scenario, "{:03d}".format(movement)]
                data = LoadData(skelColor=[1, 0, 0, 0.5])
                data.readData(LabelData_List, ['a', 'c', 'i'])
                dataLoad.append(data)

            # Condition to decide calculation of joint orientation error or segment error, 1 = segment error, 2 = joint orientation error
            if executeBlock == 1:
                # Calculation of segment error
                for segment in segmentNameList:
                    meanSquaredError = 0
                    standardDeviation = 0

                    errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs([[dataLoad[0].getSegIdxByName(segment)]], dataLoad[0], dataLoad[1])
                    # ---------------------- Manaul calculation of mean squared error - Not used ------------------------------------------------------ #
                    # for values in errSeq[0]:
                    #     meanSquaredError += (values - avgErr)**2
                    # meanSquaredError = meanSquaredError / len(errSeq[0])
                    # print("Mean Squared : ", meanSquaredError)
                    # ---------------------- Manaul calculation of mean squared error - Not used ------------------------------------------------------ #
                    meanSquaredError = statistics.variance(errSeq[0])
                    standardDeviation = statistics.stdev(errSeq[0])
                    # ---------------------------------------------------- Not used ------------------------------------------------------------------- #
                    # print("standard deviation : ", standardDeviation)
                    # print("Variance : ", statistics.variance(errSeq[0]))
                    # print("Mean Error: ", meanSquaredError)
                    # writer.writerow({subject, movement, segment, avgErr, avgErrEuler})
                    # writer.writerow({'Subject': subject, 'Movement': movement, 'Segment': segment, 'TotalAngleAvgError': avgErr[0], 'TotalEulerAvgError': avgErrEuler[0], 'MeanSquaredError': meanSquaredError, 'Standard Deviation': standardDeviation})
                    # ---------------------------------------------------- Not used ------------------------------------------------------------------- #
            elif executeBlock == 2 or executeBlock == 3 or executeBlock == 4 or executeBlock == 5 or executeBlock == 6:
                # Calculation of joint orientation error
                for pairs in segmentPairs:
                    if isinstance(segmentPairs[pairs], list):
                        # If a key has multiple values, Ex. 'RightUpperArm': ['T8', 'RightForeArm'], use each combination to proceed and find relevant information
                        for value in segmentPairs[pairs]:
                            meanSquaredError = 0
                            standardDeviation = 0
                            # Find the joint position between two segments
                            jointPosition = determineJointPositionBetween2Segemnts(pairs, value)
                            errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs([[dataLoad[0].getSegIdxByName(pairs), dataLoad[0].getSegIdxByName(value)]], dataLoad[0], dataLoad[1])
                            if executeBlock == 3 or executeBlock == 4:
                                # To write into 'AngleData.csv' and 'AngleDataDownSampled.csv'
                                makeAngleFile(subject, csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle)
                            if executeBlock == 5:
                                # To write into 'MRPDataWithSubject.csv'
                                makeMRPFile(subject, csvWriter, movement, jointPosition, MRPLooseData, MRPTightData)
                            if executeBlock == 6:
                                # Separate joints for P1 and P2, which then can be used for training and testing
                                if subject == 'P1':
                                    relevantJoints = jointListP1.get(str(movement))
                                elif subject == 'P2':
                                    relevantJoints = jointListP2.get(str(movement))
                                if jointPosition in relevantJoints:
                                    # To write into 'RelevantMRPDataWithSubject.csv'
                                    makeMRPFile(subject, csvWriter, movement, jointPosition, MRPLooseData, MRPTightData)
                            meanSquaredError = statistics.variance(errSeq[0])
                            standardDeviation = statistics.stdev(errSeq[0])
                    else:
                        # If key has single value, Ex. 'Head': 'T8',
                        meanSquaredError = 0
                        standardDeviation = 0
                        # Find the joint position between two segments
                        jointPosition = determineJointPositionBetween2Segemnts(pairs, segmentPairs[pairs])
                        errSeq, errSeqEuler, avgErr, avgErrEuler, looselyDataAngle, tightlyDataAngle, MRPLooseData, MRPTightData = computeErrSegs(
                            [[dataLoad[0].getSegIdxByName(pairs), dataLoad[0].getSegIdxByName(segmentPairs[pairs])]], dataLoad[0],
                            dataLoad[1])

                        if executeBlock == 3 or executeBlock == 4:
                            # To write into 'AngleData.csv' and 'AngleDataDownSampled.csv'
                            makeAngleFile(subject, csvWriter, jointPosition, looselyDataAngle, tightlyDataAngle)
                        if executeBlock == 5:
                            # To write into 'MRPDataWithSubject.csv'
                            makeMRPFile(subject, csvWriter, movement, jointPosition, MRPLooseData, MRPTightData)
                        if executeBlock == 6:
                            # Separate joints for P1 and P2, which then can be used for training and testing
                            if subject == 'P1':
                                relevantJoints = jointListP1.get(str(movement))
                            elif subject == 'P2':
                                relevantJoints = jointListP2.get(str(movement))
                            if jointPosition in relevantJoints:
                                # To write into 'RelevantMRPDataWithSubject.csv'
                                makeMRPFile(subject, csvWriter, movement, jointPosition, MRPLooseData, MRPTightData)
                        meanSquaredError = statistics.variance(errSeq[0])
                        standardDeviation = statistics.stdev(errSeq[0])
            # Flush dataLoad array for fresh movement
            dataLoad = []

# block below to create plots
path = ''

if executeBlock == 1:
    # Create Segments Error Plots
    readErrorData = pd.read_csv('SegmentErrorData.csv', skiprows=[0], header=None)              # Read csv into a data frame for further processing
    path = '../Plots/Error Plots/'                                                              # Define path to write graphs
    createplotsforsegments(readErrorData, path)
elif executeBlock == 2:
    # Create Joint Orientation Error Plots
    readErrorData = pd.read_csv('JointOrientationError.csv', skiprows=[0], header=None)
    path = '../Plots/Joint Orientation Error Plots/'
    createplotsforjointOrientation(readErrorData, path)