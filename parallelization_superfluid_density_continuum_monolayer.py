x# -*- coding: utf-8 -*-

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
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force, integrate_Romberg

c = 3e18 # nm/s  #3e9 # m/s
m_e =  5.1e8 / c**2 # meV s²/m²
m = 0.403 * m_e # meV s²/m²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm

Delta = 0.08   #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 8.76 #8*Delta # meV*nm    # 8 * Delta  #0.644 meV 

cut_off = 1.1*k_F # 1.1 k_F

theta = np.pi/2 #np.pi/2   # float
B_x = 0
B_y = 0
N = 514  #514   #300
n_cores = 16
points = 1* n_cores
N_polifit = 2  # 4
C = 0

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "C": C
              }

# Define the search space
search_space = [(-0.002 * k_F, 0.002 * k_F)]  # Search from -3 to 3

def function(q, q_B_x, q_B_y):
    q_x = q * np.cos(theta - np.pi/2)
    q_y = q * np.sin(theta - np.pi/2)
    integral, low_integral, high_integral = integrate_Romberg(N, mu, B_x, B_y, q_B_x, q_B_y,
                                                              Delta, gamma, Lambda, k_F, cut_off,
                                                              q_x, q_y, phi_x=0, phi_y=0)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(q_B_x**2 + q_B_y**2 + q_x**2 + q_y**2) - 2*mu + gamma*cut_off**2)
    return energy_phi

def E_0(q_B_x, q_B_y, q_x, q_y, phi_x, phi_y):
    integral, low_integral, high_integral = integrate_Romberg(N, mu, B_x, B_y, q_B_x, q_B_y,
                                                              Delta, gamma, Lambda, k_F, cut_off,
                                                              q_x, q_y, phi_x, phi_y)
    energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral) + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2 + phi_y**2 + q_x**2 + q_y**2 + q_B_x**2 + q_B_y**2) - 2*mu + gamma*cut_off**2)
    return energy_phi

def get_minima(search_space, B_x, B_y):
    def objective_function(x):
        """Objective function to minimize"""
        return function(x[0], B_x, B_y)
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=15,                    # Only 25 expensive evaluations!
        n_initial_points=5,           # Start with 10 random points
        random_state=None,
        acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
        noise=0.0,
        initial_point_generator="lhs",
        x0=[0.]
    )
    return result.x[0]
