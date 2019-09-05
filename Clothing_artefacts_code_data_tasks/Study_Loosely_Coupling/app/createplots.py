import math
from bokeh.plotting import figure, output_file, save
from bokeh.models.annotations import Title

movementList = ['', 'N-Pose', 'Three Step Calibration Palm Front', 'Three Step Calibration N-Pose', 'X Sense Gait Calibration', 'Trunk Flexion Extension', 'Truck Rotation', 'Trunk Side Bending', 'Squats', 'Upper Arm Flexion Extension', 'Upper Arm Abduction Closed Sleeve', 'Upper Arm Abduction Open Sleeve', 'Upper Arm Rotation Open Sleeve', 'Shrugging Open Sleeve', 'Lover Arm Flexion Open Sleeve', 'Lower Arm Flexion Upper Arm Flexed 90 Degree Open Sleeve', 'Lower Arm Rotation Open Sleeve', 'Picking Up from Floor - Putting to Shelf Open Sleeve', 'Picking Up from Floor - Walking - Putting Down Open Sleeve', 'Screwing Light Bulb Open Sleeve']


def saveplots(subjectName, path, meanSquaredErrorArrayForPlot, standardDeviationArrayForPlot, segmentsArrayForPlot, avgErrorArrayForPlot, counter):
    # print(subjectName)
    output_file(path + subjectName + " - " + str(counter) + ".html")  # Name of the file to save the plot
    p = figure(x_range=segmentsArrayForPlot, plot_width=800, plot_height=800)  # Create an initial figure
    # p.vbar(x=segmentsArrayForPlot, top=meanSquaredErrorArrayForPlot, width=0.9, legend='Mean Squared Error')  # Plot bar graph for mean squared error
    p.y_range.start = 0
    p.xaxis.major_label_orientation = math.pi / 2
    p.line(x=segmentsArrayForPlot, y=standardDeviationArrayForPlot, legend="Standard deviation", color='red',
           # Plot line graph for Standard deviation
           line_width=2)
    p.line(x=segmentsArrayForPlot, y=avgErrorArrayForPlot, legend="Average Error", color='green')
    # p.line(x=segmentsArrayForPlot, y=eulerAvgErrorArrayForPlot, legend="Average Euler Error", color='yellow')
    p.legend.location = "top_left"
    t = Title()
    t.text = subjectName + " - " + movementList[counter] + " : Standard deviation and Mean squared error for segments"
    p.title = t
    save(p)  # Save the plot in a file
    return
    # eulerAvgErrorArrayForPlot = []
    # # End if


def createplotsforsegments(readErrorData, path):
    meanSquaredErrorArrayForPlot = []  # Array to hold all mean squared error for a particular movement
    standardDeviationArrayForPlot = []  # Array to hold all standard deviation for a particular movement
    segmentsArrayForPlot = []  # Array to hold all segments for a particular movement
    avgErrorArrayForPlot = []  # Array to hold all average for a particular movement - Not used currently
    counter = 1  # Counter to control when to create the plot using bokeh
    for index, rowsOfReadErrorData in readErrorData.iterrows():
        if rowsOfReadErrorData[1] != counter or index == len(readErrorData) - 1:                  # Plot for the previous movement if the current movement changed. This means that all the data for the previous movement is plotted. But if it is the last movement then plot that, as we will not come in this block of code again.
            saveplots(subjectName, path, meanSquaredErrorArrayForPlot, standardDeviationArrayForPlot, segmentsArrayForPlot, avgErrorArrayForPlot, counter)
            meanSquaredErrorArrayForPlot = []  # Reset all arrays
            standardDeviationArrayForPlot = []  # Reset all arrays
            segmentsArrayForPlot = []  # Reset all arrays
            avgErrorArrayForPlot = []
        segmentsArrayForPlot.append(rowsOfReadErrorData[2])
        meanSquaredErrorArrayForPlot.append(rowsOfReadErrorData[5])
        standardDeviationArrayForPlot.append(rowsOfReadErrorData[6])
        avgErrorArrayForPlot.append(rowsOfReadErrorData[3])
        counter = rowsOfReadErrorData[1]                                                          # Assign the current movement number every time to @var : counter, to keep track of change
        subjectName = rowsOfReadErrorData[0]                                                      # Assign the subject name every time

def createplotsforjointOrientation(readErrorData, path):
    meanSquaredErrorArrayForPlot = []  # Array to hold all mean squared error for a particular movement
    standardDeviationArrayForPlot = []  # Array to hold all standard deviation for a particular movement
    segmentsArrayForPlot = []  # Array to hold all segments for a particular movement
    avgErrorArrayForPlot = []  # Array to hold all average for a particular movement - Not used currently
    counter = 1  # Counter to control when to create the plot using bokeh
    for index, rowsOfReadErrorData in readErrorData.iterrows():
        if rowsOfReadErrorData[1] != counter or index == len(readErrorData) - 1:                  # Plot for the previous movement if the current movement changed. This means that all the data for the previous movement is plotted. But if it is the last movement then plot that, as we will not come in this block of code again.
            saveplots(subjectName, path, meanSquaredErrorArrayForPlot, standardDeviationArrayForPlot, segmentsArrayForPlot, avgErrorArrayForPlot, counter)
            meanSquaredErrorArrayForPlot = []  # Reset all arrays
            standardDeviationArrayForPlot = []  # Reset all arrays
            segmentsArrayForPlot = []  # Reset all arrays
            avgErrorArrayForPlot = []
        if rowsOfReadErrorData[2] == 'Head' and rowsOfReadErrorData[3] == 'T8':
            segmentsArrayForPlot.append('Neck')
        elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'T8':
            segmentsArrayForPlot.append('Glenohumeral Right Joint')
        elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'T8':
            segmentsArrayForPlot.append('Glenohumeral Left Joint')
        elif rowsOfReadErrorData[2] == 'RightUpperArm' and rowsOfReadErrorData[3] == 'RightForeArm':
            segmentsArrayForPlot.append('Right Elbow')
        elif rowsOfReadErrorData[2] == 'LeftUpperArm' and rowsOfReadErrorData[3] == 'LeftForeArm':
            segmentsArrayForPlot.append('Left Elbow')
        elif rowsOfReadErrorData[2] == 'RightForeArm' and rowsOfReadErrorData[3] == 'RightHand':
            segmentsArrayForPlot.append('Right Wrist')
        elif rowsOfReadErrorData[2] == 'LeftForeArm' and rowsOfReadErrorData[3] == 'LeftHand':
            segmentsArrayForPlot.append('Left Wrist')
        elif rowsOfReadErrorData[2] == 'T8' and rowsOfReadErrorData[3] == 'Pelvis':
            segmentsArrayForPlot.append('T8 - Pelvis')
        elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'RightUpperLeg':
            segmentsArrayForPlot.append('Right Hip')
        elif rowsOfReadErrorData[2] == 'Pelvis' and rowsOfReadErrorData[3] == 'LeftUpperLeg':
            segmentsArrayForPlot.append('Left Hip')
        elif rowsOfReadErrorData[2] == 'RightUpperLeg' and rowsOfReadErrorData[3] == 'RightLowerLeg':
            segmentsArrayForPlot.append('Right Knee')
        elif rowsOfReadErrorData[2] == 'LeftUpperLeg' and rowsOfReadErrorData[3] == 'LeftLowerLeg':
            segmentsArrayForPlot.append('Left Knee')
        elif rowsOfReadErrorData[2] == 'RightLowerLeg' and rowsOfReadErrorData[3] == 'RightFoot':
            segmentsArrayForPlot.append('Right Ankle')
        elif rowsOfReadErrorData[2] == 'LeftLowerLeg' and rowsOfReadErrorData[3] == 'LeftFoot':
            segmentsArrayForPlot.append('Left Ankle')

        meanSquaredErrorArrayForPlot.append(rowsOfReadErrorData[6])
        standardDeviationArrayForPlot.append(rowsOfReadErrorData[7])
        avgErrorArrayForPlot.append(rowsOfReadErrorData[4])
        counter = rowsOfReadErrorData[1]                        # Assign the current movement number every time to @var : counter, to keep track of change
        subjectName = rowsOfReadErrorData[0]                    # Assign the subject name every time