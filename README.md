# Laser Beam Alignment

A trained neural network that is able to align a laser beam to the center of two irises with 50 micrometer precision.

The important files in this project are:

RasterScan.py          --> Strafes a laser over an iris with a power meter behind it to create a scan of the laser power with a certain iris
XYMoveSamples.py       --> Takes scan data files as inputs and creates an efficient training set based on them
LaserAlignment.py      --> Trains the network based on samples created by XYMoveSamples
PhysicalAlignment.py   --> Uses the trained neural network to align a laser between two irises

picoMotor.py           --> A library for all standard picoMotor functions
picoTest.py            --> A set of examples that show how the picoMotors work
STOP.py                --> Emergency cancel for the picomotors that also resets the joystick

Everything else was either used to get to this point or has a non-necessary function


