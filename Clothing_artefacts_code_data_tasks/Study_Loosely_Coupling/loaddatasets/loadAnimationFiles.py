import numpy as np
# from utils.trafos import quaternion_matrix4x4
from Clothing_artefacts_code_data_tasks.Study_Loosely_Coupling.utils.trafos import quaternion_matrix4x4

class structtype():
    pass

stateLength = 14

def loadAnimationFile(animtationFile, sampleLimit):
    aData = structtype()  # [structtype() for i in range(nTime)]
    loadAnimationFileRef(aData, animtationFile, sampleLimit)
    return aData

def loadAnimationFileRef(aData, animtationFile, sampleLimit):
    aFile = open(animtationFile, "r")
    aDataAll = []
    for line in aFile:
        rLineFloat = [float(i) for i in line.split()]
        aDataAll.append(rLineFloat)

    # create struct for animation data
    nSegs = int(len(aDataAll[0]) / stateLength)
    if (sampleLimit == -1):
        nTime = len(aDataAll)
    else:
        nTime = min(sampleLimit, aDataAll)
    seg = [structtype() for i in range(nSegs)]

    for sIdx in range(0, nSegs):
        seg[sIdx].pos_g = np.zeros((3, nTime))
        seg[sIdx].quat_sg = np.zeros((4, nTime))
        seg[sIdx].T = np.zeros((16, nTime))
        # seg[sIdx].T = np.zeros((4,4, nTime))
        T = np.zeros((4, 4))
        for t in range(nTime):
            seg[sIdx].pos_g[:, t] = aDataAll[t][sIdx * stateLength:sIdx * stateLength + 3]
            seg[sIdx].quat_sg[:, t] = aDataAll[t][sIdx * stateLength + 3:sIdx * stateLength + 7]
            T = quaternion_matrix4x4(seg[sIdx].quat_sg[:, t])
            T[0:3, 3] = seg[sIdx].pos_g[:, t]
            # seg[sIdx].T[:,:,t] = quaternion_matrix4x4(seg[sIdx].quat_sg[:,t])
            # seg[sIdx].T[0:3,3,t] = seg[sIdx].pos_g[:,t]
            seg[sIdx].T[:, t] = T.reshape(16)
            # seg[sIdx].pos_si  = aDataAll[t][sIdx * stateLength + 7:sIdx * stateLength + 10]
            # seg[sIdx].quat_si = aDataAll[t][sIdx * stateLength + 10:sIdx * stateLength + 14]
    aData.seg = seg
    aData.nTime = nTime
    aData.nSegs = nSegs