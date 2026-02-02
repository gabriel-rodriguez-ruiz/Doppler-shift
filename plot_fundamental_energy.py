#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:56:10 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_folder = Path("Data/")
name = "Fundamental_energy_q_B_0.06283185307179587_q_x_in_(-0.0628-0)_mu_S_-38.0_L=100_h=0.0001_B=0.0_Delta=0.2_lambda=0.0_w_s=10_w_S=10_w_1=0.25_points=19.npz"
file_to_open = data_folder / name

data = np.load(file_to_open)
E = data["E"]
# phi_x_values = data["phi_x_values"]
q_x_values = data["q_x_values"]
q_B_x_values = data["q_B_x_values"]
Delta_s = data["Delta_s"]
Delta_S = data["Delta_S"]

fig, ax = plt.subplots()
ax.plot(q_x_values, E[:, 0, 0])


# ax.set_xlabel(r"$\phi$")
ax.set_xlabel(r"$q_x/\pi$")

ax.set_ylabel(r"$E$")
ax.legend()
ax.set_title(r"$q_{B}=$" + f"{q_B_x_values[0]}"+
             r"; $\Delta_s=$" + f"{Delta_s}"
             + r"; $\Delta_S=$" + f"{Delta_S}")

fig.tight_layout()
