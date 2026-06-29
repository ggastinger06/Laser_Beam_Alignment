# Laser Beam Alignment

A trained neural network that is able to align a laser beam to the center of two irises with 15 micrometer precision.

## The important files in this project are:

### RasterScan.py          
Drives the picomotors in a raster scan in order to get data to generate samples from

### XYMoveSamples.py       
Creates a set of samples to train on given a set of raster scans

### LaserAlignment.py      
Trains the model with data from XYMoveSamples in order to find ideal weights

### PhysicalAlignment.py   
Rebuilds the model given a weights file and uses the model to continously improve laser position until the convergence condition is met

### picoMotor.py          
A small library that contains common picoMotor functions

### picoTest.py            
A set of examples that show how the picoMotors can be used

### STOP.py                
Emergency stop for the picomotors that restores all hardware to a safe state



