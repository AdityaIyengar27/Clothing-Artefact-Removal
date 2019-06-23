class structtype():
    pass

def loadCalibFile(calibFile, nSegs):
    cData = structtype()
    loadCalibFileRef(cData, calibFile, nSegs)
    return cData

def loadCalibFileRef(cData, calibFile, nSegs):
    file = open(calibFile, "r")
    names = {}
    vnames = {}
    segCalib = [structtype() for i in range(0,nSegs)]
    n=0
    for line in file:
        listLine = line.split()
        if (n<nSegs):
            names[listLine[0]]=n
            vnames[n] = listLine[0]
            segCalib[n].pos_si = [float(i) for i in listLine[1:4]]
            segCalib[n].quat_si = [float(i) for i in listLine[4:8]]
            n += 1
        else:
            nPoints = int((len(listLine)-1)/3)
            points=[]
            for p in range(0,nPoints):
                points.append([float(i) for i in listLine[1+3*p:1+3*p+3]])
            segCalib[names[listLine[0]]].points = points
    cData.segCalib = segCalib
    cData.nSegs = nSegs
    cData.nIMUs = nSegs # could be changed in future!
    cData.segValueByNames = names
    cData.segNamesByValue = vnames
