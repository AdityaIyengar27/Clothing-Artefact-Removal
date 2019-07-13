import numpy as np
# from loaddatasets.loadAnimationFiles import loadAnimationFile, loadAnimationFileRef
# from loaddatasets.loadCalibrationFiles import loadCalibFile, loadCalibFileRef
# from loaddatasets.loadIMUData import loadIMUData, loadIMUDataRef
# from utils.trafos import quaternion_matrix, quationion_to_MRP, MRP_to_quaternion
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.loaddatasets.loadAnimationFiles import loadAnimationFile, loadAnimationFileRef
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.loaddatasets.loadCalibrationFiles import loadCalibFile, loadCalibFileRef
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.loaddatasets.loadIMUData import loadIMUData, loadIMUDataRef
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.utils.trafos import quaternion_matrix, quationion_to_MRP, MRP_to_quaternion

import os

class structtype():
    pass

class LoadData():
    def __init__(self, skelColor=[1,1,1,0,5]):
        self.aData = structtype()
        self.cData = structtype()
        self.iData = structtype()
        self.Id = ""
        self.skelColor = skelColor

    def obtainFileName(self, data_List, attribute):
        pathData = data_List[1] + "/InputData/" + data_List[0] + "/"
        listFiles = os.listdir(pathData)
        fileName = []
        for fname in listFiles:
            if ((attribute in fname[0]) and (data_List[-1] in fname) and (data_List[-2] in fname)):
                fileName.append(fname)
        # print(fileName)
        return fileName

    def resetLoader(self):
        self.__init__()

    def readData(self, data_List, dataLoad = ['a', 'c', 'i'], sampleLimit=-1):
        pathData =  data_List[1] + "/InputData/" + data_List[0] + "/"
        idata = [structtype() for i in range(0,23)]

        if ('a' in dataLoad):
            # Load Animation files
            animationFile = self.obtainFileName(data_List, 'a')
            if (len(animationFile) == 1):
                file = pathData + animationFile[0]
                loadAnimationFileRef(self.aData, file, sampleLimit)
            else:
                raise Exception('Error Matching filenames: ' + str(len(animationFile)))
        if ('c' in dataLoad):
            # Load Calibration data
            calibFile = self.obtainFileName(data_List, 'c')
            if (len(calibFile) == 1):
                file = pathData + calibFile[0]
                loadCalibFileRef(self.cData, file, self.aData.nSegs)
            else:
                raise Exception('Error Matching filenames: ' + str(len(calibFile)))
        if ('i' in dataLoad):
            # Load IMU Data
            imuFile = self.obtainFileName(data_List, 'i')
            if (len(imuFile) == 1):
                file = pathData + imuFile[0]
                # loadIMUDataRef(self.iData, file, self.aData.nSegs, self.cData.segValueByNames, sampleLimit)
                idata = loadIMUData(file, self.aData.nSegs, self.cData.segValueByNames, sampleLimit)
            else:
                raise Exception('Error Matching filenames: ' + str(len(imuFile)))

    def getQuatSegmentByName(self, segName, t):
        return self.aData.seg[self.cData.segValueByNames[segName]].quat_sg[:, t]

    def getQuatSegment(self, segIdx, t):
        return self.aData.seg[segIdx].quat_sg[:, t]

    def setQuatSegment(self, segIdx, t, quat):
        self.aData.seg[segIdx].quat_sg[:, t] = quat

    def getMRPSegmentByName(self, segName, t):
        return quationion_to_MRP(self.aData.seg[self.cData.segValueByNames[segName]].quat_sg[:, t])

    def getMRPSegment(self, segIdx, t):
        return quationion_to_MRP(self.aData.seg[segIdx].quat_sg[:, t])

    def getPosSegmentByName(self, segName, t):
        return self.aData.seg[self.cData.segValueByNames[segName]].pos_g[:, t]

    def getTransformSegmentByName(self, segName, t):
        return self.aData.seg[self.cData.segValueByNames[segName]].T[:, t]

    def getSegIdxByName(self, segName):
        return self.cData.segValueByNames[segName]

    def getSegNameByIdx(self, segIdx):
        return self.cData.segNamesByValue[segIdx]

    def getArrayExportOfSequence(self):
        # exporting orientations as MRPs (3-vector) for each segment and each time-step
        lVar = 3
        exportArray = np.zeros((self.aData.nTime, len(self.aData.seg)*lVar))
        for t in range(self.aData.nTime):
            for segIdx in range(len(self.aData.seg)):
                exportArray[t, segIdx*lVar:segIdx*lVar + lVar] = self.getMRPSegment(segIdx, t)
        return exportArray

    def setQuatStatesFromArray(self, importArray):
        lVar = 3
        for t in range(self.aData.nTime):
            for segIdx in range(len(self.aData.seg)):
                quat = MRP_to_quaternion(importArray[t, segIdx*lVar:segIdx*lVar + lVar])
                self.setQuatSegment(segIdx, t, quat)


def loadDataLoosely(pathLearningData, scenarioLabel, sampleLimit=-1, dataLoad = ['a', 'c', 'i']):
    animtationFile = pathLearningData +'a'+ scenarioLabel+'.dat'
    calibFile = pathLearningData +'c'+scenarioLabel+'.dat'
    imuFile = pathLearningData +'i'+scenarioLabel+'.dat'

    aData = []
    cData = []
    iData = []

    if ('a' in dataLoad):
        # Load Animation files
        aData = loadAnimationFile(animtationFile, sampleLimit)
    if ('c' in dataLoad):
        # Load Calibration data
        cData = loadCalibFile(calibFile, aData.nSegs)
    if ('i' in dataLoad):
        # Load IMU Data
        iData = loadIMUData(imuFile, aData.nSegs, cData.segValueByNames, sampleLimit)
    return aData, cData, iData
