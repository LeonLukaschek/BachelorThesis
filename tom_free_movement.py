"""
    This file implements the free-moving/free-rotating TOM-CubeSat model.
"""

from simulatorModelChesi import SimulatorModel_Chesi
from simulatorModelChesi_ADAPTIVE import SimulatorModel_Chesi_ADAPTIVE

from simulationProperties import SimulationProperties

import time, sys

import numpy as np
import matplotlib.pyplot as plt

from kf_cm_approximation import * 

from plotter import plot_data

np.set_printoptions(threshold=sys.maxsize) 

## Rotation matrices & transformations
def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])

def rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

def rotation_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def cad_to_body_matrix(J):
    """ Transforms a matrix from the cad frame to the body frame.
    """
    R = rotation_x(np.pi/2)

    J1 = np.dot(R, np.dot(J, R.T))

    R2 = rotation_z(np.pi/2)

    return np.dot(R2, np.dot(J1, R2.T))

def cad_to_body_vector(v):
    """ Transforms a vector from the cad frame to the body frame
    """
    R1 = rotation_x(np.pi/2)

    v1 = np.dot(R1, v)

    R2 = rotation_z(np.pi/2)

    return np.dot(R2, v1)

def main():
    # The simulation settings to be used. JSON file has to be placed into the simulation_properties folder.
    sim_set = SimulationProperties(file="tom.json")

    ## Transformation of simulation settings into right frames
    # Transform inertia into sim frame
    # sim_set.J0 = cad_to_body_matrix(J=sim_set.J0)
    # # Transform r_true into sim frame
    # sim_set.r_true = cad_to_body_vector(sim_set.r_true)

    print(sim_set.toStr())
    print("")
    t_start = time.time()

    ## Initialisation of the system model with the loaded properties
    chesi_simulator = SimulatorModel_Chesi(J0=sim_set.J0, m=sim_set.m, g=sim_set.g, R_true=sim_set.r_true)

    ## Creating a array of timesteps
    t = np.arange(sim_set.dt, sim_set.tf+sim_set.dt, sim_set.dt)

    ## Creating the initial model state 
    y = np.hstack((sim_set.attitude_initial, sim_set.angular_vel_initial, sim_set.r_true))

    ## Adding initial state to array of all states
    data = [y]
    
    ##########################
    ## Main simulation loop ##
    ##########################
    
    for i in range(0, len(t)-1):
        # simulates one step and saves the resulting y vector (roll, pitch, yaw, wx, wy, wz) into the data list
        curr_step = chesi_simulator.step(y=data[i], dt=sim_set.dt)
        data.append(curr_step) 
        
    ## Extraction of the euler-angles and angular velocities from the simulated data
    angles = [elem[0:3] for elem in data]
    angular_vel = [elem[3:6] for elem in data]

    ## Creating the euler-angles and angular velocities plots
    plot_data(angles=angles, angular_vel=angular_vel, t=t, plots=["attitude & angular_vel"], sim_set=sim_set)                
         
    ## Debugging information
    t_end = time.time()
    print("________________________________")
    print(f"\nRuntime: {round(t_end-t_start,4)}s")



if __name__ == "__main__":
    main()