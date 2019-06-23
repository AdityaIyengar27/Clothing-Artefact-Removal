# Visualization, Evaluation and Data Export and Import for Skeleton data of loosely and tightly coupled BSN recordings
This folder contains data and code (based on Python 3) to visualize and evaluate the recorded data.
# Installation Ubuntu (18.04 LTS)
To install pip and python3, use the following commands in the bash (terminal):
(Python3)

```
$ sudo apt-get install python3
```

(Pip for python3)

```
$ sudo apt-get update
$ sudo apt-get install python3-pip
```

The code is completely Python3 based and depends on several libraries that can be install via pip as follows:

```
$ sudo python3 -m pip install pyquaternion pyqtgraph PyOpenGL PyOpenGL_accelerate PyQt5 numpy matplotlib 
```

# Structur of Folders and Usage
The "app" folder contains the main script "main.py".
The "data" folder contains all the data and "InputData" and "Labels" of the different motion scenarios that where recorded.
The "datavisualization", "loaddatasets", and "utils" folder contain utility functions and classes.

# Usage
You can execute the script "main.py", e.g. via the terminal:

```
$ cd app
$ python3 main.py
```

The script contains different sections that can be configured:

* Load data: 
Subjects can be chosen (we have "P1" and "P2") and movement numbers that select the movement data.
Note: the formated numbers select the respective file in the "data/InputData" folder
and the numbers coinside with the movement labels in folder "data/Labels" 
(line-number in "MoevementLabels_..." files, matches "MovementNr" in script) 

* Online Error plots: 
A list of segments can be provided to show error curves of 
the respective segments (orienation-error (scalar value) + euler splitting (x,y,z-axis) as (R,G,B)), 
between the tracked segments from the tightly / lossely coupled capturing alongside 
with the 3D animation of the motion.

* Executing a Task: 
A list of strings can be provided to either 
	* visualize the data (alongside the online error-plots)
	* evaluate the orientation difference of the two setups on the given data-sequence
	* write-out the orienation-sequence of the skeleton for sequence-to-sequence learning (numpy array of all timesteps in this sequence)

