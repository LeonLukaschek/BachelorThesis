"""
    This file implements the KF-based CM approximation of the TOM-CubeSat model.
    If SIM_RUNS is set to more than 1, the average offset estimate and offset error over all runs will be calculated.
"""

from simulatorModelChesi import SimulatorModel_Chesi
from simulatorModelChesi_ADAPTIVE import SimulatorModel_Chesi_ADAPTIVE

from simulationProperties import SimulationProperties

from printer import vector_to_str_pretty

import time, sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kf_cm_approximation import * 

from plotter import plot_data

np.set_printoptions(threshold=sys.maxsize) 

# Number of simulation runs
SIM_RUNS = 1
counter = 0
list_of_estimates = list()

# For usinig data by optitrack system
USE_OPTI_DATA = True

OPTI_DATA_FILE = "opti_data/small_initial_angular_vel.csv"


df = pd.DataFrame

# Used for calculating the average of estimate/estimate errors of multiple simulation runs
r_estimate_avg_total = []
r_estimate_error_avg_total = [np.array([0, 0, 0])]
r_estimate_error_avg_total_mm = [np.array([0, 0, 0])]

# rotation matrices
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

def opti_to_body_matrix(J):
    """ Transforms a matrix from the cad frame to the body frame.
    """
    R1 = rotation_y(-np.pi/2)
    J1 = np.dot(R1, np.dot(J, R1.T))

    R2 = rotation_x(np.pi/2)
    J2 = np.dot(R2, np.dot(J1, R2.T))

    return J2

def opti_to_body_vector(v):
    """ Transforms a vector from the cad frame to the body frame
    """
    R1 = rotation_y(np.pi/2)
    v1 = np.dot(R1, v)

    R2 = rotation_x(np.pi/2)
    v2 = np.dot(R2, v1)

    return v2

def main():
    global r_estimate_avg_total
    global r_estimate_error_avg_total
    global counter
    global list_of_estimates

    ## Loading of the simulation properties
    sim_set = SimulationProperties(file="mass-sim.json")
    # Transform inertia into sim/kf frame
    # sim_set.J0 = cad_to_body_matrix(J=sim_set.J0)
    # Transform r_true into sim/kf frame
    # sim_set.r_true = cad_to_body_vector(sim_set.r_true)
    # Transform r_guess into sim/kf frame
    # sim_set.r_guess = cad_to_body_vector(sim_set.r_guess)



    if USE_OPTI_DATA:
        print(f"Running KF approximation using data: {OPTI_DATA_FILE}")
        print(sim_set.toStr())
    else:
        print(f"Running simulation using KF")
        print(sim_set.toStr())

    t_start = time.time()


    ## Initialisation of the system model with the loaded properties
    chesi_simulator_KF = SimulatorModel_Chesi(J0=sim_set.J0, m=sim_set.m, g=sim_set.g, R_true=sim_set.r_true)
        
    ## Creating a array of timesteps
    t = np.arange(sim_set.dt, sim_set.tf+sim_set.dt, sim_set.dt)
    y = None

    if not USE_OPTI_DATA:
        ## Creating the initial model state 
        y = np.hstack((sim_set.attitude_initial, sim_set.angular_vel_initial, sim_set.r_true))
        
        ## Adding initial state to array of all states
        data = [y]

    ## Initialization of the KF approximator
    cm_kf = CM_approximation_KF(R_true=sim_set.r_guess)

    ## Creation of a initial state
    init_r_guess = sim_set.r_guess 
    cm_estimate = [np.hstack((sim_set.angular_vel_initial, init_r_guess))]


    ##########################
    ## Main simulation loop ##
    ##########################
        
    if not USE_OPTI_DATA:
        for i in range(0, len(t)-1):
            # simulates one step and saves the resulting y vector (roll, pitch, yaw, wx, wy, wz) into the data list
            curr_step = chesi_simulator_KF.step(y=data[i], dt=sim_set.dt)
            data.append(curr_step) 

    if USE_OPTI_DATA:
        df = pd.DataFrame(pd.read_csv(OPTI_DATA_FILE))

        data_opti = list()

        for k in range(0, len(df["Time (Seconds)"])):
            # (roll, pitch, yaw, omega_x, omega_y, omega_z, r_x, r_y, r_z)
            # -> data from optitrack has to be transformed because the opti and sim coord frames dont match
            # -> could be done by transformation matrices or just by changing the order of euler angles
            att = np.array((df.iloc[k]["roll rad"], df.iloc[k]["pitch rad"], df.iloc[k]["yaw rad"]))

            ang_vel = np.array((df.iloc[k]["omega_x rad"], df.iloc[k]["omega_y rad"], df.iloc[k]["omega_z rad"]))

            # Adapted opti data that fits to KF
            new_entry = np.array((att[0], att[1], att[2],
                                  ang_vel[0], ang_vel[1], ang_vel[2],
                                  sim_set.r_true[0], sim_set.r_true[1], sim_set.r_true[2]))
                
            data_opti.append(new_entry)

    ########################
    ## Kalman Filter Loop ##
    ########################
    length = len(t)
    if USE_OPTI_DATA:
        length = len(df["Time (Seconds)"])

    for k in range(1, length):
        # data = [roll, pitch, yaw , omega_x, omega_y, omega_z, rx, ry, rz]
        atttitude_km1 = None
        atttitude_k = None 

        if USE_OPTI_DATA:
            atttitude_km1 = data_opti[k-1][0:3]
            atttitude_k = data_opti[k][0:3]
        else:
            atttitude_km1 = data[k-1][0:3]
            atttitude_k = data[k][0:3]
        phi_mat = cm_kf.calc_phi_matrix(dT=sim_set.dt, m=sim_set.m, g=sim_set.g, J=sim_set.J0, attitude_km1=atttitude_km1, attitude_k=atttitude_k)
                
        F = cm_kf.calc_F_k(phi_mat=phi_mat)
                
        # x = np.transpose(np.transpose(data[k][3:]) + np.sqrt(cm_kf.Q).dot(np.random.randn(6, )).flatten())
        x = None
        z = None
        if USE_OPTI_DATA:
            x = np.transpose(data_opti[k][3:])
            z = cm_kf.H.dot(x) # no additional noise as opti data is noisy
        else:
            x = np.transpose(np.transpose(data[k][3:]) + np.sqrt(cm_kf.Q).dot(np.array([np.random.random(), np.random.random(), np.random.random(), np.random.random(), np.random.random(), np.random.random()])))
            z = cm_kf.H.dot(x) + np.sqrt(cm_kf.R).dot(np.random.randn(3,1))
            # fixing dimensions to be (3,) instead of (3,1)
            z_corr = np.array((z[0,0], z[1,0], z[2,0]))
            z = z_corr
        ## Kalman filter steps
        P_pri = F.dot(cm_kf.P_post).dot(np.transpose(F)) + cm_kf.Q
                
        K = P_pri.dot(np.transpose(cm_kf.H)).dot(np.linalg.inv(cm_kf.H.dot(P_pri).dot(np.transpose(cm_kf.H)) + cm_kf.R))
        # KalG.append(np.linalg.norm(K))
        # Prediction phase
        xhat_pri = F.dot(cm_kf.xhat_post)
        # Correction phase
        cm_kf.xhat_post = xhat_pri + K.dot(z-cm_kf.H.dot(xhat_pri))
        cm_kf.xhat_post = np.array((cm_kf.xhat_post[0,0], cm_kf.xhat_post[0,1], cm_kf.xhat_post[0,2], cm_kf.xhat_post[0,3], cm_kf.xhat_post[0,4], cm_kf.xhat_post[0,5]))
        cm_kf.P_post = (np.eye(6) - K.dot(cm_kf.H)).dot(P_pri.dot(np.transpose(np.eye(6)-K.dot(cm_kf.H)))) + K.dot(cm_kf.R).dot(np.transpose(K))
        cm_estimate.append(cm_kf.xhat_post)

    if USE_OPTI_DATA:
        data = data_opti
        t = df["Time (Seconds)"]
    plot_data(title="", plots = ["cm_estimate"], angles=[elem[0:3] for elem in data], angular_vel=[elem[3:6] for elem in data], t=t, cm_estimate=cm_estimate, sim_set=sim_set)

    # Calculates and prints the avg of the last len(data)*X% estimated offset vectors
    r_avg = cm_kf.calc_R_avg(data=cm_estimate, samples=len(t)*0.15)
    r_error = sim_set.r_true - r_avg

    r_estimate_avg_total.append(r_avg)
    r_estimate_error_avg_total.append(r_error)

    r_true_formatted = vector_to_str_pretty(sim_set.r_true, 10)
    r_estimate_formatted = vector_to_str_pretty(r_avg, 10)
    r_error_formatted = vector_to_str_pretty(r_error, 10)
    r_error_mm = vector_to_str_pretty(r_error * 10**(3), 10)

    print(f"KF: ")
    print(f"\tTrue offset vector:\t\t[{r_true_formatted[0]}|{r_true_formatted[1]}|{r_true_formatted[2]}]")
    print(f"\tOffset vector estimate:\t\t[{r_estimate_formatted[0]}|{r_estimate_formatted[1]}|{r_estimate_formatted[2]}]")
    print(f"\tOffset vector error:\t\t[{r_error_formatted[0]}|{r_error_formatted[1]}|{r_error_formatted[2]}]")
    print(f"\tOffset vector error [mm]:\t[{r_error_mm[0]}|{r_error_mm[1]}|{r_error_mm[2]}]")
     
    
    t_end = time.time()

    print("_____________________________________")
    print(f"\nRuntime: {round(t_end-t_start,4)}s")


if __name__ == "__main__":
    main()