"""
    This file implements the adaptive-control-based CM approximation of the TOM-CubeSat model.
    If SIM_RUNS is set to more than 1, the average offset estimate and offset error over all runs will be calculated.
"""

from simulatorModelChesi import SimulatorModel_Chesi
from simulatorModelChesi_ADAPTIVE import SimulatorModel_Chesi_ADAPTIVE

from simulationProperties import SimulationProperties

from printer import vector_to_str_pretty

import time, sys

import numpy as np
import matplotlib.pyplot as plt

from kf_cm_approximation import * 

from plotter import plot_data

np.set_printoptions(threshold=sys.maxsize) 

CM_APPROXIMATOR = "ADAPTIVE_CONTROL"

# Used for calculating the average of estimate/estimate errors of multiple simulation runs
r_estimate_avg_total = [np.array([0, 0, 0])]
r_estimate_error_avg_total = [np.array([0, 0, 0])]
r_estimate_error_avg_total_mm = [np.array([0, 0, 0])]


def main():
    global r_estimate_avg_total
    global r_estimate_error_avg_total

    print(f"Running simulation using {CM_APPROXIMATOR}")

    ## Loading of the simulation properties
    sim_set = SimulationProperties(file="tom.json")

    # For tom simulation properties 1.5kg are added for support structure (for mmus etc)
    if sim_set.file == "tom.json":
        sim_set.m += 1.5

    print(sim_set.toStr())
    print("")
    t_start = time.time()


    results_transversal = list()
    results_vertical = list()

    ################
    ## SIMULATION ##
    ################

    ## Model used for the adaptive control esimator
    sim_adaptive = SimulatorModel_Chesi_ADAPTIVE(sim_set.k_p, sim_set.dt, sim_set.v_mmu, sim_set.m_mmu, sim_set.m,
                                                         sim_set.J0, sim_set.g, sim_set.tf, sim_set.r_true, sim_set.r_guess,
                                                         sim_set.mmu_min, sim_set.mmu_max)

    ## Simulates the satellite simulator using equations of motion and estimates the x-y components of the  offset vector.
    ## Moves movable masses at each simulation step to correct CM offset.
    # results_transversal = [tArray, omegaArray, r_mmusArray, r_estArray, ctrltorqueArray]
    results_transversal = sim_adaptive.transversalCorrection()
    print("Transversal correction done.")

    ## Estimation of the z component of the offset vector using UKF
    # results_vertical =  [tArray, attitudeArray, angular_velArray, zhatArray]
    results_vertical = sim_adaptive.verticalCorrection()
    print("Vertical correction done.")
                  
    ################
    ## EVALUATION ##
    ################
            
    ## Calculate the average of the x/y/z offset vector estimates by averaging the last 50% of estimated values
    xy_transversal = np.array(results_transversal[3])
    transversal_rmmus = np.array(results_transversal[2])

    z_vertical = np.array(results_vertical[3])
    x_avg, y_avg, z_avg = 0.0, 0.0, 0.0
    sample_fraction = 0.10
    x_avg = np.sum(xy_transversal[int(len(xy_transversal[:, 0])*sample_fraction):, 0])/(len(xy_transversal[:, 0]) - int(len(xy_transversal[:, 0])*sample_fraction))
    y_avg = np.sum(xy_transversal[int(len(xy_transversal[:, 1])*sample_fraction):, 1])/(len(xy_transversal[:, 1]) - int(len(xy_transversal[:, 1])*sample_fraction))
    z_avg = np.sum(z_vertical[int(len(z_vertical)*sample_fraction):])/(len(z_vertical) - int(len(z_vertical)*sample_fraction))

    estimate_adaptive_avg = np.array((x_avg, y_avg, z_avg))
    r_adaptive_error = sim_set.r_true - estimate_adaptive_avg

    r_estimate_avg_total.append(estimate_adaptive_avg)
    r_estimate_error_avg_total.append(r_adaptive_error)
            

    r_true_formatted = vector_to_str_pretty(sim_set.r_true, 10)
    r_estimate_formatted = vector_to_str_pretty(estimate_adaptive_avg, 10)
    r_error_formatted = vector_to_str_pretty(r_adaptive_error, 10)
    r_error_mm = vector_to_str_pretty(r_adaptive_error * 10**(3), 10)

    ## Information about estimate print
    print(f"Adpative control:")
    print(f"\tTrue offset vector:\t\t[{r_true_formatted[0]}|{r_true_formatted[1]}|{r_true_formatted[2]}]")
    print(f"\tOffset vector estimate:\t\t[{r_estimate_formatted[0]}|{r_estimate_formatted[1]}|{r_estimate_formatted[2]}]")
    print(f"\tOffset vector error:\t\t[{r_error_formatted[0]}|{r_error_formatted[1]}|{r_error_formatted[2]}]")
    print(f"\tOffset vector error [mm]:\t[{r_error_mm[0]}|{r_error_mm[1]}|{r_error_mm[2]}]")  
    print("________________________________")          


    ## Plotting information about the transversal phase            
    plot_data(t=results_transversal[0], angular_vel=results_transversal[1], transversal_rmmus=transversal_rmmus, plots=["transversal_rmmus"], cm_estimate=results_transversal[3], sim_set=sim_set)
    plot_data(t=results_transversal[0], angular_vel=results_transversal[1], plots=["cm_estimate_adaptive_xy"], cm_estimate=results_transversal[3], sim_set=sim_set)
    # Plots information about the vertical phase
    plot_data(t=results_vertical[0], angles=results_vertical[1], angular_vel=results_vertical[2], plots=["cm_estimate_adaptive_z"], cm_estimate=results_vertical[3], sim_set=sim_set)
    # plot_data(t=results_vertical[0], angles=results_vertical[1], angular_vel=results_vertical[2], plots=["attitude"], cm_estimate=results_vertical[3], sim_set=sim_set, info_str=sim_set.toStr(), show_plt=True)

    t_end = time.time()

    
    print(f"\nRuntime: {round(t_end-t_start,4)}s")

if __name__ == "__main__":
    main()