# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:44:11 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
from functions import get_energy, get_energy_in_polars
    
#%% Parameters
k_y = [0.*np.pi]
phi_x = [0.*np.pi]
phi_y = [0]
q_B_x = np.array([0.006*np.pi])
q_B_y = [0.]
q_x = [0]
q_y = [0]
w_s = 10   #10
w_S = w_s #w_s/3  #20
Delta_s = 0   #0
Delta_S = 0.2
# E_0 = 25.3333 #25.33  #25.3846
mu_s = -3.8*w_s #-3.8*w_s  #-38
mu_S =  w_S/w_s * mu_s  #-38 + E_0 #-38 + E_0
theta = np.pi/2 #np.pi/2
B = 0*Delta_S  #0.8*Delta_S #0 * Delta_S    #2 * Delta_S
B_x = B * np.cos(theta) 
B_y = B * np.sin(theta)
g = 0 #1/5
B_x_S =  g * B * np.cos(theta) 
B_y_S =  g * B * np.sin(theta) 
Lambda = 0  #0.5  #0.5   #0.5
w_1 = [0.25]  # 0.25

#%% Plot energy bands

fig, ax = plt.subplots()
for w in w_1:
    for i in range(8):
        L_x = 1000
        k_x = np.pi*np.arange(-L_x, L_x)/L_x
        Energy = get_energy(k_x, k_y, phi_x, phi_y, w_s, w_S,
                            mu_s, mu_S, Delta_s, Delta_S, B_x,
                            B_y, B_x_S, B_y_S, Lambda, w, q_B_x, q_B_y,
                            q_x, q_y)
        ax.plot(k_x/np.pi, Energy[:, 0, 0, 0, 0, 0, 0, 0, i], label=f"{w}")


fig.suptitle(r"$\lambda=$" + f"{np.round(Lambda,2)}"
             +r"; $\Delta=$" + f"{Delta_S}"
             + r"; $\mu_s=$"+f"{np.round(mu_s, 3)}"
             + r"; $\mu_S=$"+f"{np.round(mu_S, 3)}"
             + r"; $w_s=$"+f"{np.round(w_s, 3)};"
             + "\n"
             + r"$w_S=$"+ f"{np.round(w_S, 3)}"
             +r"; $B_x=$"+f"{np.round(B_x, 2)}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $w_1=$" + f"{w_1}" + r"; $q_x=$" + f"{q_x}"
             + r"; $q_B/\pi=$" + f"{q_B_x/np.pi}")
plt.tight_layout()
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$E(k_x, k_y=0)$")
# ax.set_xlim(-0.6, 0.6)
# ax.set_ylim(-0.2, 0.2)

plt.grid()
plt.tight_layout()


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
