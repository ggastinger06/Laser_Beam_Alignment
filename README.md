# Laser Beam Alignment

Summer internship project: automated laser beam alignment using Picomotor actuators, a Moku device for power readout, and a neural network for alignment prediction.

## Contents

- **Moku/** — Moku and Picomotor control scripts. `3.0 RasterScan.py` performs raster scans of the beam; `BasicTesting/` has device connection tests and an emergency `STOP.py`; `Archive/` holds earlier scan versions and result plots.
- **Neural Network/** — TensorFlow models that predict alignment moves from power readings. `MultiAdjust/` is the current multi-axis alignment code with real scan data (`scan_data_*.npz` from raster scans); `Old_Models/` and `Examples/` are earlier and reference implementations.
- `PicoMotor.pdf` — Picomotor actuator documentation.
- `Driver standoff.stl` — 3D-printable mounting standoff.

## Setup

Requires Python 3.11. Each subproject uses its own virtual environment (`Moku/.venv`, `Neural Network/moku-env`); key packages are `moku`, `tensorflow`, `numpy`, and `matplotlib`.
