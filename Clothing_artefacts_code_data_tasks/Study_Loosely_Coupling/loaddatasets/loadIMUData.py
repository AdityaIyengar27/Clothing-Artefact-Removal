import numpy as np

class structtype():
    pass

def loadIMUData(imuFile, nIMUs, segNames, sampleLimit):
    iData = [structtype() for i in range(0, nIMUs)]
    loadIMUDataRef(iData, imuFile, nIMUs, segNames, sampleLimit)
    return iData

def loadIMUDataRef(iData, imuFile, nIMUs, segNames, sampleLimit):
    file = open(imuFile, "r")
    imuData = [structtype() for i in range(0, nIMUs)]
    # iData = [structtype() for i in range(0, nIMUs)]
    iDataTmp = []
    nLoc=0
    t=0
    nIMU=0
    for line in file:
        listLine = line.split()
        #print listLine
        if (len(listLine)==1):
            nIMU=int(listLine[0])
            imuData = [structtype() for i in range(0, nIMU)]
        else:
            idx = segNames[listLine[0]]
            # if(idx == nIMU):
            #     imuData[nLoc].name = listLine[0]
            #     imuData[nLoc].t = float(listLine[1])
            #     imuData[nLoc].acc = [float(i) for i in listLine[2:5]]
            #     imuData[nLoc].gyr = [float(i) for i in listLine[5:8]]
            #     imuData[nLoc].mag = [float(i) for i in listLine[8:11]]
            #     imuData[nLoc].q_gi = [float(i) for i in listLine[11:15]]
            # else:
            imuData[nLoc].name = listLine[0]
            imuData[nLoc].t    = float(listLine[1])
            imuData[nLoc].acc  = [float(i) for i in listLine[2:5]]
            imuData[nLoc].gyr  = [float(i) for i in listLine[5:8]]
            imuData[nLoc].mag  = [float(i) for i in listLine[8:11]]
            imuData[nLoc].q_gi = [float(i) for i in listLine[11:15]]
            nLoc+=1
        if (nLoc==nIMU):
            iDataTmp.append(imuData)
            nLoc=0
        if (sampleLimit > -1 and len(iDataTmp) == sampleLimit):
            break

    nTime = len(iDataTmp)
    for imu in range(nIMU):
        iData[imu].name = iDataTmp[0][imu].name
        iData[imu].t = np.zeros(nTime)
        iData[imu].acc = np.zeros((3,nTime))
        iData[imu].gyr = np.zeros((3,nTime))
        iData[imu].mag = np.zeros((3,nTime))
        iData[imu].q_gi = np.zeros((4,nTime))

    for imu in range(nIMU):
        for t in range(len(iDataTmp)):
            iData[imu].t[t] = iDataTmp[t][imu].t
            iData[imu].acc[:,t] = iDataTmp[t][imu].acc
            iData[imu].gyr[:, t] = iDataTmp[t][imu].gyr
            iData[imu].mag[:, t] = iDataTmp[t][imu].mag
            iData[imu].q_gi[:, t] = iDataTmp[t][imu].q_gi