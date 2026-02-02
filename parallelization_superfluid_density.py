# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from skopt import gp_minimize
from skopt.space import Space
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')
import scipy
from pathlib import Path
from functions import get_fundamental_energy

L_x = 100  #300
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
h = 1e-4
phi_x_values = [0]
phi_y_values = [0]
q_x_values = [0]
q_y_values = [0]
q_B_x_values = np.array([0.01*np.pi])
q_B_y_values = [0]
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_calls = 15
n_initial_points = 5
# Define the search space
search_space = [(-0.02 * np.pi, 0.02 * np.pi)]  
n_cores = 19
points = 1* n_cores

parameters = {"L_x": L_x, "L_y": L_y, "w_s": w_s, "w_S": w_S,
          "mu_s": mu_s, "mu_S": mu_S, "Delta_s": Delta_s, "Delta_S": Delta_S, "theta": theta,
           "Lambda": Lambda,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "w_s": w_s, "w_S": w_S, "w_1":w_1,
          "g": g, "q_x_values": q_x_values,
          "q_y_values": q_y_values, "q_B_x_values": q_B_x_values,
          "q_B_y_values": q_B_y_values}

def function(q, q_B_x, q_B_y):
    q_x = q * np.cos(theta + np.pi/2)
    q_y = q * np.sin(theta + np.pi/2)
    energy_q = get_fundamental_energy(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
                           B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values=phi_x_values, phi_y_values=phi_y_values,
                           q_B_x_values=[q_B_x], q_B_y_values=[q_B_y],
                           q_x_values=[q_x], q_y_values=[q_y])[0, 0]    
    return energy_q

def E_0(q_x, q_y, q_B_x, q_B_y):
    energy_q = get_fundamental_energy(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
                           B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values=phi_x_values, phi_y_values=phi_y_values,
                           q_B_x_values=[q_B_x], q_B_y_values=[q_B_y],
                           q_x_values=[q_x], q_y_values=[q_y])[0, 0]
    return energy_q

def get_minima(search_space, q_B_x, q_B_y):
    def objective_function(x):
        """Objective function to minimize"""
        return function(x[0], q_B_x, q_B_y)
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=n_calls,                    # Only 25 expensive evaluations!
        n_initial_points=n_initial_points,           # Start with 10 random points
        random_state=None,
        acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
        noise=0.0,
        initial_point_generator="lhs",
        x0=[[-q_B_x_values[0]], [0.], [q_B_x_values[0]]]
    )
    return result.x[0]

def integrate_q_B(q_B):
    q_B_x = q_B * np.cos(theta + np.pi/2)
    q_B_y = q_B * np.sin(theta + np.pi/2)
    q_eq = get_minima(search_space, q_B_x, q_B_y)
    phi_x_values = np.array([-h, 0, h]) + q_eq * np.cos(theta + np.pi/2)
    phi_y_values = np.array([-h, 0, h]) + q_eq * np.sin(theta + np.pi/2)
    superfluid_density_xx = 1/(L_x*L_y) * (E_0(phi_x_values[2], phi_y_values[1], q_B_x, q_B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], q_B_x, q_B_y) + E_0(phi_x_values[0], phi_y_values[1], q_B_x, q_B_y))/h**2
    superfluid_density_xx_0 = 1/(L_x*L_y) * (E_0(h, 0, q_B_x, q_B_y) - 2*E_0(0, 0, q_B_x, q_B_y) + E_0(-h, 0, q_B_x, q_B_y))/h**2
    superfluid_density_yy = 1/(L_x*L_y) * (E_0(phi_x_values[1], phi_y_values[2], q_B_x, q_B_y) - 2*E_0(phi_x_values[1], phi_y_values[1], q_B_x, q_B_y) + E_0(phi_x_values[1], phi_y_values[0], q_B_x, q_B_y))/h**2
    superfluid_density_yy_0 = 1/(L_x*L_y) * (E_0(0, h, q_B_x, q_B_y) - 2*E_0(0, 0, q_B_x, q_B_y) + E_0(0, -h, q_B_x, q_B_y))/h**2
    superfluid_density_xy = 1/(L_x*L_y) * (E_0(phi_x_values[2], phi_y_values[2], q_B_x, q_B_y) - E_0(phi_x_values[2], phi_y_values[0], q_B_x, q_B_y) - E_0(phi_x_values[0], phi_y_values[2], q_B_x, q_B_y) + E_0(phi_x_values[0], phi_y_values[0], q_B_x, q_B_y))/(4*h**2)
    superfluid_density_xy_0 = 1/(L_x*L_y) * (E_0(h, h, q_B_x, q_B_y) - E_0(h, -h, q_B_x, q_B_y) - E_0(-h, h, q_B_x, q_B_y) + E_0(-h, -h, q_B_x, q_B_y))/(4*h**2)
    return q_eq, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0

if __name__ == "__main__":
    q_B_values = np.linspace(0.*np.pi, 0.01*np.pi, points)
    B_direction = f"{theta:.2}"
    integrate = integrate_q_B
    with multiprocessing.Pool(n_cores) as pool:
        q_eq, superfluid_density_xx, superfluid_density_xx_0, superfluid_density_yy, superfluid_density_yy_0, superfluid_density_xy, superfluid_density_xy_0 = zip(*pool.map(integrate, q_B_values))
        q_eq = np.array(q_eq)
        superfluid_density_xx = np.array(superfluid_density_xx)
        superfluid_density_xx_0 = np.array(superfluid_density_xx_0)
        superfluid_density_yy = np.array(superfluid_density_yy)
        superfluid_density_yy_0 = np.array(superfluid_density_yy_0)
        superfluid_density_xy = np.array(superfluid_density_xy)
        superfluid_density_xy_0 = np.array(superfluid_density_xy_0)
        data_folder = Path("Data/")
        name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(q_B_values/np.pi),3)}-{np.round(np.max(q_B_values/np.pi),3)})_phi_x_in_({np.round(np.min(phi_x_values), 3)}-{np.round(np.max(phi_x_values),3)})_Delta_S={Delta_S}_Delta_s={Delta_s}_lambda={np.round(Lambda, 2)}_points={points}_points={points}_N={L_x}_w_S={w_S}.npz"
        file_to_open = data_folder / name
        np.savez(file_to_open,
                 superfluid_density_xx = superfluid_density_xx,
                 superfluid_density_xx_0 = superfluid_density_xx_0,
                 superfluid_density_yy = superfluid_density_yy,
                 superfluid_density_yy_0 = superfluid_density_yy_0,
                 superfluid_density_xy = superfluid_density_xy,
                 superfluid_density_xy_0 = superfluid_density_xy_0,
                 q_B_values = q_B_values, q_eq = q_eq, **parameters)
        print("\007")