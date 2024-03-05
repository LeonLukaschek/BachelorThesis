# Bachelor Thesis - Algorithmic center of mass offset approximation of a spherical air-bearing satellite simulator

This repo contains the python code used for my bachelor thesis. In it, different approaches are tested to approximate the Center of Rotation (CoR) to Center of Mass (CoM) offset.
It is structured as follows:

Files starting with *tom_* are the main entry points to the different simulations.

*tom_free_movement.py* simulates the free movement of the TOM-CubeSat on the attitude simulator. 

*tom_KF.py* simulates the CubeSat and approximates the CoM with a (basic) Kalman Filter. 

*tom_adaptive_control.py* simulates the CubeSat and approximates the CoM with two phase, adaptive control algorithm consisting of a control based and an UKF based 
sequence.

## How to run

### *tom_free_movement.py*
For this, you have to executed the python file using 
```console
python3 tom_free_movement.py
```
It will simulate the dynamics of the attitude simulator and plot the angular velocities and attitude.

### *tom_KF.py*
For this, you have to executed the python file using 
```console
python3 tom_KF.py
```
It will simulate the dynamics of the attitude simulator as well as approximate the CoM with a KF. After that, it will plot the angular velocities and attitude.
If the parameter *USE_OPTI_DATA* in line 31 is set to *TRUE*, the Kalman Filter will run using a specified OptiTrack file. The OptiTrack file is specified in line 33, 
as *OPTI_DATA_FILE*. The folder *opti_data* contains recorded data sets similar to the data sets in the thesis. 

### *tom_adaptive_control.py*
For this, you have to executed the python file using 
```console
python3 tom_adaptive_control.py
```
It will simulate the dynamics of the attitude simulator as well as approximate the CoM using an adaptive control based law. It works in two steps.
The first step approximates the X- and Y-Component of the CoM offset vector using moveable masses nullifying the CoM offset iteratively. 
Following, the Z-Component will be estimated by an unscented Kalman Filter. After that, three plots will open: first the attitude and angular velocites, then the 
X- and Y-CoM-offset-components, then the Z-CoM-offset-component.

### General informations
The *tom_*-files plot their results using the helper tool *plotter.py*. It contains matplotlib code fitting to the different plot scenarios.

Also, the simulation properties are supplied by JSON-files. Different simulation settings can be found in the folder *simulation_properties*. The usage of JSON files for simulations 
allows the quick change of values without changing the simulation code itself. The contents of the thesis were all generated/constructed using the file *tom.json*. In each *tom_* file, 
the simulation properties are loaded by the following line
```python
    sim_set = SimulationProperties(file="tom.json")
```

If there are any issues or proplems, you can contact me at [leon.lukaschek@gmail.com](mailto:leon.lukaschek@gmail.com).
