# sysid-multinoise

## System identification for linear systems with multiplicative noise

The code in this repository implements the algorithms and ideas from our paper:
* Linear System Identification under Multiplicative Noise from Multiple Trajectory Data(under review)


## Dependencies
* Python 3.5+ (tested with 3.7.3)
* NumPy
* SciPy
* Matplotlib

## Installing
Currently there is no formal package installation procedure; simply download this repository and run the Python files.

## Examples
There are several example Python files which can be run.

### demo_script_lite.py
Use this file as a "Hello world" for this repository. The program should perform system identification of a small 2-state, 1-input system with multiplicative noise. The program should run 4 experiment trials, produce a plot of the model estimate error vs # of rollouts, save the results and plot image, then terminate. 

### demo_script.py
This script was used to run the experiment and generate the figure presented in the paper.


## General code structure
The core model parameter estimation code is located in "system_identification.py". Various simulated experiments can be run by the functions in "experiments.py". Example linear dynamic system definitions are located in "system_definitions.py". Plotting functions are provided in "plotting.py". Utility functions are located in "matrixmath.py", "pickle_io.py", and "utility.py".

## General code notes
Experiment results are automatically saved into the "experiments" folder under a subfolder named by the time when the experiment was ran (the "timestr" variable in the code).
Printed output can be suppressed by setting the "print_updates" function input to "False".


## Authors
* **Ben Gravell** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
* **Yu Xing** - [Chinese Academy of Sciences]
* **Xingkang He** - [KTH](https://people.kth.se/~xingkang/index.html)
* **Tyler Summers** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
* **Karl Johannsson** - [KTH](https://people.kth.se/~kallej/)
