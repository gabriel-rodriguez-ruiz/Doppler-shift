# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:37:26 2024

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from functions import get_fundamental_energy

L_x = 300
L_y = L_x
w_s = 10   #10
w_S = w_s  #10/3
w_1 = 0.25
Delta_s = 0 # 0 ###############Normal state
Delta_S = 0.2
# E_0 = 25.3846  #26.6154   #25.3846
mu_s = -3.8*w_s   #-3.8*w_s   #-3.8*w_s
mu_S = w_S/w_s * mu_s
theta = np.pi/2
g = 0   #1/5
B = 0 * Delta_S #0.8 * Delta_S
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
B_x_S = g * B * np.cos(theta)
B_y_S = g * B * np.sin(theta)
Lambda = 0.  #0.5
phi_x_values = [0]
phi_y_values = [0]
q_x_values = [0]
q_y_values = [0]
q_B_x_values = [0.001*np.pi]
q_B_y_values = [0]
h = 1e-4
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
n_cores = 19    # odd to include 0
points = 1*n_cores


params = {"L_x": L_x, "L_y": L_y, "w_s": w_s, "w_S": w_S,
          "mu_s": mu_s, "mu_S": mu_S, "Delta_s": Delta_s, "Delta_S": Delta_S, "theta": theta,
           "Lambda": Lambda,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "w_s": w_s, "w_S": w_S, "w_1":w_1,
          "g": g, "B": B, "theta": theta, "B_x": B_x, "B_y": B_y}


def integrate_q_x(q_x):
    E = get_fundamental_energy(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
                           B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values=phi_x_values, phi_y_values=phi_y_values,
                           q_B_x_values=q_B_x_values, q_B_y_values=q_B_y_values,
                           q_x_values=[q_x], q_y_values=q_y_values)
    return E


if __name__ == "__main__":
    q_x_values = np.linspace(-0.02*np.pi, 0.02*np.pi, points)

    with multiprocessing.Pool(n_cores) as pool:
        # results_pooled = pool.map(integrate, phi_x_values)
        results_pooled = pool.map(integrate_q_x, q_x_values)
    E = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"Fundamental_energy_q_B_{q_B_x_values[0]}_q_x_in_({np.round(np.min(q_x_values), 4)}-{np.round(np.max(q_y_values), 4)})_mu_S_{mu_S}_L={L_x}_h={np.round(h,4)}_B={B}_Delta={Delta_S}_lambda={Lambda}_w_s={np.round(w_s, 3)}_w_S={np.round(w_S, 3)}_w_1={np.round(w_1, 3)}_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , E=E, q_B_x_values=q_B_x_values, q_B_y_values=q_B_y_values,
             q_x_values=q_x_values, q_y_values=q_y_values,
             phi_x_values=phi_x_values, phi_y_values=phi_y_values,
             **params)
    
    print("\007")