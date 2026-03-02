# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:44:11 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from diagonalization import get_Energies_in_polars
    
#%% Parameters
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
Lambda = 0#8.76 #8*Delta # meV*nm    # 8 * Delta  #0.644 meV 

cut_off = 1.1*k_F # 1.1 k_F

theta = -np.pi/2 #np.pi/2   # float
B_x = 0
B_y = 0
N = 150  #514   #300
n_cores = 8
points = 1* n_cores
n_calls = 15
n_initial_points = 5
h = 1e-4

q_B_x = 0
q_B_y = 0.001
q_x = 0
q_y = 0
phi_x = 0
phi_y = 0
parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "Lambda": Lambda, "N": N,
              "cut_off": cut_off
              }

#%% Plot energy bands

fig, ax = plt.subplots()
L_x = 1000
k_x = np.pi*np.linspace(0.02, 0.03, L_x)
Energy = get_Energies_in_polars(k_x, [theta], B_x, B_y,
                               q_B_x, q_B_y,
                               mu, Delta, gamma, Lambda,
                               q_x, q_y,
                               phi_x, phi_y)

ax.plot(k_x, Energy[:, 0, 0])
ax.plot(k_x, Energy[:, 0, 1])
ax.plot(k_x, Energy[:, 0, 2])
ax.plot(k_x, Energy[:, 0, 3])



#%% Plot pockets in the bilayer

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

r_min = 0
r_max = 0.2*np.pi
N = 200
radius_values = np.linspace(r_min, r_max, N)
theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
radius, theta = np.meshgrid(radius_values, theta_values)

Energies_polar = get_energy_in_polars(radius_values, theta_values, phi_x, phi_y, w_s, w_S,
                           mu_s, mu_S, Delta_s, Delta_S, B_x,
                           B_y, B_x_S, B_y_S, Lambda, w_1[0], q_x, q_y)

Energies_polar = Energies_polar[:, :, 0, 0, 0, 0, :]

contours = []
for i in range(8):
    values = Energies_polar[:,:, i].T
    contour = ax.contour(theta, radius, values, levels=[0.0], colors=f"C{i}")
    contours.append(contour)

values = Energies_polar[:,:, 4].T
# Create masks for positive and negative values
mask_positive = values >= 0
mask_negative = values < 0

# Plot negative values in another color
ax.scatter(theta[mask_negative], radius[mask_negative], color='red', label='Negative Values',
           s=1)
ax.set_title(r"$B/\Delta=$" + f"{np.round(B/Delta_S, 3)}")

#%% Plot fundamental energy
