import numpy as np
from pyquaternion import Quaternion
from utils.trafos import euler_from_quaternion
import matplotlib.pyplot as plt

class Evaluation():
    def __init__(self):
        self.ToDeg = 180 / np.pi

    def qRel(self, q1, q2):
        return q1.inverse* q2

    def scalarAngleErr(self, qRel):
        value = max(-1.0,min(qRel.elements[0], 1.0))
        value = 2 * np.arccos(value)*self.ToDeg
        if(value > 180):
            value = abs(360 - value)
        return value

    def angleErr(self, q1, q2):
        return self.scalarAngleErr(self.qRel(q1, q2))

    def angleErrEuler(self, q1, q2):
        qRel = self.qRel(q1, q2)
        errAngles = np.asarray(euler_from_quaternion(np.asarray(qRel.elements), axes='sxyz'))
        return np.abs(errAngles)*self.ToDeg

    def angleErrRel(self, q1, q2, q3, q4):
        qRel1 = self.qRel(q1, q2)
        qRel2 = self.qRel(q3, q4)
        return self.scalarAngleErr(self.qRel(qRel1, qRel2))

    def angleErrEulerRel(self, q1, q2, q3, q4):
        qRel1 = self.qRel(q1, q2)
        qRel2 = self.qRel(q3, q4)
        errAngles = np.asarray(euler_from_quaternion(np.asarray(self.qRel(qRel1, qRel2).elements), axes='sxyz'))
        return np.abs(errAngles)*self.ToDeg


def computeErrSegs(errSegs, dataLoosely, dataTightly):
    err = Evaluation()
    avgErr = []
    avgErrEuler = []
    errSeq = []
    errSeqEuler = []
    nTime = min(dataTightly.aData.nTime, dataLoosely.aData.nTime)
    for i in range(len(errSegs)):
        avgErr.append(0)
        avgErrEuler.append(np.zeros(3))
        errSeq.append(np.linspace(0, 0, nTime))
        errSeqEuler.append([np.linspace(0, 0, nTime) for l in range(3)])
        for t in range(nTime):
            if (len(errSegs[i]) == 1):
                # print("Segment idx: " + str(segIdx[0]))
                qTight_sg = Quaternion(dataTightly.getQuatSegment(errSegs[i][0], t))
                qLoose_sg = Quaternion(dataLoosely.getQuatSegment(errSegs[i][0], t))
                totalAngleErr = err.angleErr(qTight_sg, qLoose_sg)
                eulerAngles = err.angleErrEuler(qTight_sg, qLoose_sg)

            if (len(errSegs[i]) == 2):
                qT_sg0 = Quaternion(dataTightly.getQuatSegment(errSegs[i][0], t))
                qT_sg1 = Quaternion(dataTightly.getQuatSegment(errSegs[i][1], t))
                qL_sg0 = Quaternion(dataLoosely.getQuatSegment(errSegs[i][0], t))
                qL_sg1 = Quaternion(dataLoosely.getQuatSegment(errSegs[i][1], t))
                totalAngleErr = err.angleErrRel(qT_sg0, qT_sg1, qL_sg0, qL_sg1)
                eulerAngles = err.angleErrEulerRel(qT_sg0, qT_sg1, qL_sg0, qL_sg1)

            errSeq[-1][t] = totalAngleErr
            # print("Error in deg: " + str(totalAngleErr) + " at time: " + str(t))
            for l in range(3):
                errSeqEuler[-1][l][t] = eulerAngles[l]
            # print("Error in deg if axes (x,y,z): " + str(eulerAngles) + " at time: " + str(t))
            avgErr[-1] += totalAngleErr
            avgErrEuler[-1] += eulerAngles

        avgErr[-1] /= nTime
        # if(avgErr[-1] > 180):
        #     avgErr[-1] = abs(360 - avgErr[-1])
        # avgErrEuler[-1] /= nTime
        # print("Average total error: " + str(avgErr[-1]))
        # print("Average error per axis (x,y,z): " + str(avgErrEuler[-1]))
    return errSeq, errSeqEuler, avgErr, avgErrEuler

def plotErrSegs(loader, errSegs, errSegsSeq):
    rows = 1
    cols = len(errSegs)
    if (len(errSegs) > 2):
        rows = np.ceil(np.sqrt(len(errSegs)))
        cols = rows
    errSeq, errSeqEuler, avgErr, avgErrEuler = errSegsSeq
    nTime = len(errSeq[0])
    plt.figure(0)
    for l in range(1,len(errSegs)+1):
        plt.subplot(rows, cols, l)
        plt.ylabel("Err (deg)")
        if (len(errSegs[l-1]) == 1):
            titleStr = loader.getSegNameByIdx(errSegs[l-1][0])
        else:
            titleStr = loader.getSegNameByIdx(errSegs[l - 1][0]) + '_' + loader.getSegNameByIdx(errSegs[l - 1][1])
        plt.title(titleStr)
        plt.plot(np.linspace(0, nTime, nTime), errSeq[l-1][:], 'k--')
        plt.plot(np.linspace(0, nTime, nTime), errSeqEuler[l-1][0][:], 'r')
        plt.plot(np.linspace(0, nTime, nTime), errSeqEuler[l-1][1][:], 'g')
        plt.plot(np.linspace(0, nTime, nTime), errSeqEuler[l-1][2][:], 'b')
    plt.show()
