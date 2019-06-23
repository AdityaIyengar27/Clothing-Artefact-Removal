from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui
import sys
import pyqtgraph.opengl as gl
from pyqtgraph import Transform3D
from datavisualization.customClasses import GLAxisItemOwn, SegmentItem
import numpy as np
import pyqtgraph as pg

from utils.evaluationMethods import Evaluation, Quaternion

class Viewer():
    def __init__(self, errSegs = []):
        self.app = QApplication([])
        pg.setConfigOption('background', 'w')
        self.view = gl.GLViewWidget()
        #self.view.setBackgroundColor('w')
        self.view.show()
        # create floor
        self.grid = gl.GLGridItem()
        self.view.addItem(self.grid)
        #self.grid.rotate(90, 0, 1, 0)  # obtain z-up config
        # set initial camera config
        #self.view.setCameraPosition(distance=10.0, elevation=-60, azimuth=0)
        #self.view.pan(dx=0, dy=0, dz=2)

        # create global origin
        self.origin = GLAxisItemOwn(size=QtGui.QVector3D(0.5,0.5,0.5))
        self.view.addItem(self.origin)
        self.skels = []
        self.aData = []
        self.errSegs = errSegs


    # def createStore(self, errSegs):
    #     dataStore = []
    #     for err in errSegs:
    #         dataStore.append(np.zeros(4))
    #     return dataStore

    def start(self):
        self.app.exec_()

    def getOrigin(self):
        return self.origin

    def setErrorPlots(self):
        self.err = Evaluation()
        #self.storeData = self.createStore(self.errSegs)
        self.visErr = pg.GraphicsWindow(title="Online error plots")

        self.visErr.resize(800, 600)
        self.visErr.setWindowTitle('Online error plots')
        self.windowWidth = 100  # width of the window displaying the curve

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
        self.pl = []
        self.curves = []
        self.Xm = []
        self.storeDataT = []
        self.ptr = -self.windowWidth
        for p in range(len(self.errSegs)):
            if (len(self.errSegs[p]) == 1):
                segName = self.skels[0].getSegNameByIdx(self.errSegs[p][0])
            else:
                segName = self.skels[0].getSegNameByIdx(self.errSegs[p][0]) + "_" + self.skels[0].getSegNameByIdx(self.errSegs[p][1])
            self.pl.append(self.visErr.addPlot(title=segName, pen='y'))
            self.pl[-1].setYRange(0,30)
            XmLoc = []
            for l in range(4):
                XmLoc.append(np.linspace(0, 0, self.windowWidth))  # create array that will contain the relevant time series
            self.Xm.append(XmLoc)
            storeDataTmp = np.zeros(4)
            self.storeDataT.append(storeDataTmp)
            self.curves.append([self.pl[p].plot(pen='y'), self.pl[p].plot(pen=(255, 0, 0), name="Red curve", width=2.)
                               , self.pl[p].plot(pen=(0, 255, 0), name="Green curve", width=2.),
                            self.pl[p].plot(pen=(0, 0, 255), name="Blue curve", width=2.)])

            if ((len(self.errSegs) > 2)):
                rSize = np.ceil(np.sqrt(len(self.errSegs)))
                if ((p+1) % rSize == 0):
                    self.visErr.nextRow()


    def setSkeleton(self, skel):
        self.skels.append(skel)
        for n in range(len(skel.segs)):
            self.view.addItem(self.skels[-1].segs[n])

        if (len(self.errSegs) > 0):
            self.setErrorPlots()


    def update(self):
        for skel, aData in zip(self.skels, self.aData):
            if (self.t < aData.nTime):
                skel.update(aData, self.t)
                print("Skel: " + skel.Id + " Timestep: " + str(self.t) + " of total: " + str(aData.nTime))

        if ((self.t < self.aData[0].nTime) and (len(self.aData) > 1) and (len(self.errSegs) > 0)):
            for i in range(len(self.errSegs)):
                if (len(self.errSegs[i]) == 1):
                    #print("Segment idx: " + str(segIdx[0]))
                    qTight_sg = Quaternion(self.aData[0].seg[self.errSegs[i][0]].quat_sg[:, self.t])
                    qLoose_sg = Quaternion(self.aData[1].seg[self.errSegs[i][0]].quat_sg[:, self.t])
                    self.storeDataT[i][0] = self.err.angleErr(qTight_sg, qLoose_sg)
                    self.storeDataT[i][1:4] = self.err.angleErrEuler(qTight_sg, qLoose_sg)
                    #print(self.storeDataT[i][:])
                if (len(self.errSegs[i]) == 2):
                    qT_sg0 = Quaternion(self.aData[0].seg[self.errSegs[i][0]].quat_sg[:, self.t])
                    qT_sg1 = Quaternion(self.aData[0].seg[self.errSegs[i][1]].quat_sg[:, self.t])
                    qL_sg0 = Quaternion(self.aData[1].seg[self.errSegs[i][0]].quat_sg[:, self.t])
                    qL_sg1 = Quaternion(self.aData[1].seg[self.errSegs[i][1]].quat_sg[:, self.t])
                    self.storeDataT[i][0] = self.err.angleErrRel(qT_sg0, qT_sg1, qL_sg0, qL_sg1)
                    self.storeDataT[i][1:4] = self.err.angleErrEulerRel(qT_sg0, qT_sg1, qL_sg0, qL_sg1)

                for j in range(4):
                    self.Xm[i][j][:-1] = self.Xm[i][j][1:]
                    self.Xm[i][j][-1] = self.storeDataT[i][j]
                    self.curves[i][j].setData(self.Xm[i][j])
                    self.curves[i][j].setPos(self.ptr,0)
            self.ptr += 1

        self.view.show()
        self.t += 1
        QtCore.QTimer.singleShot(1, self.update)

    def animate(self, list_aData):
        self.t = 0
        self.aData = list_aData
        self.update()
        self.start()

class Segments:
    pass

class DrawSkeleton():
    def __init__(self, cData, colorVec = [0,1, 1, 0.5], origin=[], nameId = "NoName"):
        self.segs = [Segments() for i in range(cData.nSegs)]
        self.Trafos = [Transform3D() for i in range(cData.nSegs)]
        self.Id = nameId
        self.cData = cData
        self.noPtsList = ['RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftShoulder',
                          'LeftUpperArm', 'LeftForeArm', 'LeftHand']
        for n in range(cData.nSegs):
            print(cData.segNamesByValue[n])
            if (cData.segNamesByValue[n] in self.noPtsList):
                listPoints = []
            else:
                listPoints = cData.segCalib[n].points
            self.segs[n] = SegmentItem(colorVec=colorVec, size=QtGui.QVector3D(0.1,0.1,0.1), listEndPts=listPoints)
            self.segs[n].setParent(origin)


    def update(self, aData, t):
        for n in range(len(self.segs)):
            self.segs[n].setTransform(aData.seg[n].T[:,t])

    def getSegments(self):
        return self.segs

    def getSegIdxByName(self, segName):
        return self.cData.segValueByNames[segName]

    def getSegNameByIdx(self, segIdx):
        return self.cData.segNamesByValue[segIdx]


def drawSkelQt(list_Loader, errSegs = []):
    global dataStore
    list_aData = []
    list_cData = []
    list_names = []
    list_skelcolors = []
    for loader in list_Loader:
        list_aData.append(loader.aData)
        list_cData.append(loader.cData)
        list_names.append(loader.Id)
        list_skelcolors.append(loader.skelColor)

    view = Viewer(errSegs)

    # create skeletons
    for cData, colorVec, name in zip(list_cData, list_skelcolors, list_names):
        skel = DrawSkeleton(cData, colorVec, view.getOrigin(), nameId=name)
        view.setSkeleton(skel)


    # go through sequence and update skeleton
    view.animate(list_aData)
