import time

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def plot_data(angles=None, angular_vel=None, t=None, plots=None, cm_estimate=None, sim_set=None, transversal_rmmus=None, title=""):

    if 'attitude & angular_vel' in plots and angles is not None:
        
        angles = np.array(angles)
        angular_vel = np.array(angular_vel)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # smoothes the graph by reducing huge jumps when values go from -0.001 -> 2PI or other way around
        unwrapped_angles = np.apply_along_axis(np.unwrap, axis=0, arr=angles.T).T

        # Euler angles
        axs[0].plot(t, unwrapped_angles[:, 0], 'r', label='roll ϕ')
        axs[0].plot(t, unwrapped_angles[:, 1], 'g', label='pitch θ')
        axs[0].plot(t, unwrapped_angles[:, 2], 'b', label='yaw ψ')
        axs[0].set_xlabel('Time (s)', fontsize=16)
        axs[0].set_ylabel('Attitude [rad]', fontsize=16)
        axs[0].legend(fontsize=16)

        axs[1].plot(t, angular_vel[:, 0], 'r', label='$ω_{x}$')
        axs[1].plot(t, angular_vel[:, 1], 'g', label='$ω_{y}$')
        axs[1].plot(t, angular_vel[:, 2], 'b', label='$ω_{z}$')
        axs[1].set_xlabel('Time (s)', fontsize=16)
        axs[1].set_ylabel('Angular Velocity [rad/s]', fontsize=16)
        axs[1].legend(fontsize=16)
       

    if 'attitude & angular_vel silva' in plots and angles is not None:
        angles = np.array(angles)
        angular_vel = np.array(angular_vel)

        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))

        # smoothes the graph by reducing huge jumps when values go from -0.001 -> 2PI or other way around
        unwrapped_angles = np.apply_along_axis(np.unwrap, axis=0, arr=angles.T).T

        if title == "":
            axs.set_title("Attitude simulator simulation evaluation - DaSilva parameters", fontsize=15)
        else:
            fig.suptitle(title, fontsize = 15)


        # Euler angles
        axs.plot(t, unwrapped_angles[:, 0], label='roll ϕ')
        axs.plot(t, unwrapped_angles[:, 1], label='pitch θ')
        axs.plot(t, unwrapped_angles[:, 2], label='yaw ψ')
        axs.set_xlabel('Time (s)', fontsize=13)
        axs.set_ylabel('Attitude [rad]', fontsize=13)
        axs.legend(fontsize=13)

    if 'attitude' in plots and angles is not None:
        angles = np.array(angles)
        # angles = angles[:-1]

        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))

        # smoothes the graph by reducing huge jumps when values go from -0.001 -> 2PI or other way around
        unwrapped_angles = np.apply_along_axis(np.unwrap, axis=0, arr=angles.T).T

        axs.set_title("Attitude simulator - simulated behaviour attitude", fontsize=20)

        # Euler angles
        axs.plot(t, unwrapped_angles[:, 0], label='roll ϕ')
        axs.plot(t, unwrapped_angles[:, 1], label='pitch θ')
        axs.plot(t, unwrapped_angles[:, 2], label='yaw ψ')
        axs.set_xlabel('Time (s)', fontsize=13)
        axs.set_ylabel('attitude [rad]', fontsize=13)
        axs.legend()

    if 'angular_vel' in plots and angular_vel is not None:
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))

        angular_vel = np.array(angular_vel)
        angular_vel = angular_vel[:-1]

        axs.set_title("Attitude simulator - simulated behaviour angular velocity", fontsize=20)

        # Angular velocities
        axs.plot(t, angular_vel[:, 0], 'r', label='$ω_{x}$')
        axs.plot(t, angular_vel[:, 1], 'g', label='$ω_{y}$')
        axs.plot(t, angular_vel[:, 2], 'b', label='$ω_{z}$')
        axs.set_xlabel('Time (s)', fontsize=13)
        axs.set_ylabel('angular velocity [rad/s]', fontsize=13)
        axs.legend(fontsize=13)

    if 'cm_estimate' in plots and cm_estimate is not None and sim_set is not None:
        fig_cm, axs_cm = plt.subplots(3, 1, figsize=(8, 6))

        cm_estimate = np.array(cm_estimate)
        cm_estimate = [elem[3:6] for elem in cm_estimate]
        cm_estimate = np.array(cm_estimate)

        axs_cm[0].plot(t[:], [sim_set.r_true[0]*10**3 for _ in cm_estimate[:]], 'k', label='$true_{x}$')
        axs_cm[0].plot(t[:], cm_estimate[:, 0]*10**3, 'r--', label='$est_{x}$')
        axs_cm[0].set_ylabel('Offset [mm]', fontsize=16)
        axs_cm[1].plot(t[:], [sim_set.r_true[1]*10**3 for _ in cm_estimate[:]], 'k', label='$true_{y}$')
        axs_cm[1].plot(t[:], cm_estimate[:, 1]*10**3, 'g--', label='$est_{y}$')
        axs_cm[1].set_ylabel('Offset [mm]', fontsize=16)
        axs_cm[2].plot(t[:], [sim_set.r_true[2]*10**3 for _ in cm_estimate[:]], 'k', label='$true_{z}$')
        axs_cm[2].plot(t[:], cm_estimate[:, 2]*10**3, 'b--', label='$est_{z}$')
        axs_cm[2].set_ylabel('Offset [mm]', fontsize=16)
        axs_cm[2].set_xlabel('Time [s]', fontsize=16)
        axs_cm[0].legend(fontsize=16)
        axs_cm[1].legend(fontsize=16)
        axs_cm[2].legend(fontsize=16)

    if 'cm_estimate_adaptive_xy' in plots and cm_estimate is not None and sim_set is not None:
        fig_cm, axs_cm = plt.subplots(2, 1, figsize=(8, 6))

        cm_estimate = np.array(cm_estimate)

        axs_cm[0].set_title("Center of Mass Estimate - Transversal Phase", fontsize=20)
        axs_cm[0].plot(t[:], [sim_set.r_true[0] for _ in cm_estimate[:]], 'k', label='$true_{x}$')
        axs_cm[0].plot(t[:], cm_estimate[:, 0], 'r--', label='$est_{x}$')
        axs_cm[0].legend(fontsize=16)
        axs_cm[0].set_ylabel('Offset (m)', fontsize=16)

        axs_cm[1].plot(t[:], [sim_set.r_true[1] for _ in cm_estimate[:]], 'k', label='$true_{y}$')
        axs_cm[1].plot(t[:], cm_estimate[:, 1], 'g--', label='$est_{y}$')
        axs_cm[1].legend(fontsize=13)
        axs_cm[1].set_ylabel('Offset (m)', fontsize=16)
        axs_cm[1].set_xlabel('Time (s)', fontsize=16)


    if 'cm_estimate_adaptive_z' in plots and cm_estimate is not None and sim_set is not None:
        fig_cm, axs_cm = plt.subplots(1, 1, figsize=(8, 6))

        cm_estimate = np.array(cm_estimate)

        axs_cm.set_title("Center of Mass Z-Component Estimate - Vertical Phase", fontsize=20)
        axs_cm.plot(t[:], [sim_set.r_true[2] for _ in range(0, len(cm_estimate))], 'k', label='$true_{z}$')
        axs_cm.plot(t[:], [elem for elem in cm_estimate], 'r--', label='$est_{z}$')
        axs_cm.set_xlabel('Time (s)', fontsize=13)
        axs_cm.set_ylabel('Offset (m)', fontsize=13)
        axs_cm.legend(fontsize=13)

    if 'transversal_rmmus' in plots and transversal_rmmus is not None and t is not None:
        #r_mmus while transversal correction
        fig_trans_rmmus, ax_trans_rmmus = plt.subplots(figsize=(8, 6))

        ax_trans_rmmus.set_title("Position of the Moveable Masses while in Transversal Correction Phase", fontsize=20)
        ax_trans_rmmus.plot(t, [value[0] for value in transversal_rmmus], color="r", label='$\mathrm{mmu}_{x}$')
        ax_trans_rmmus.plot(t, [value[1] for value in transversal_rmmus], color="g", label='$\mathrm{mmu}_{y}$')
        ax_trans_rmmus.plot(t, [value[2] for value in transversal_rmmus], color="b", label='$\mathrm{mmu}_{z}$')

        ax_trans_rmmus.set_ylabel('Position [m]', fontsize=13)
        ax_trans_rmmus.set_xlabel('Time (s)', fontsize=13)
        ax_trans_rmmus.legend(fontsize=13)
    
    
    plt.tight_layout()
    plt.show()