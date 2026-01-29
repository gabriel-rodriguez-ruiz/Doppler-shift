#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:55:22 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_B_in_1.6_(0.0-0.127)_phi_x_in_(0-0)_Delta_S=0.2_Delta_s=0_lambda=0.0_points=19_points=19.npz"


Data = np.load(file_to_open)
q_B_values = Data["q_B_values"]
Delta_S = Data["Delta_S"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_xx_0 = Data["superfluid_density_xx_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_yy = Data["superfluid_density_yy"]
superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
superfluid_density_xy = Data["superfluid_density_xy"]
superfluid_density_xy_0 = Data["superfluid_density_xy_0"]
fig, axs = plt.subplots(2, 1)

axs[0].plot(q_B_values/Delta_S, superfluid_density_xx, "o")
# axs[0].plot(q_B_values/Delta_S, superfluid_density_xx_0, "o")
# axs[0].plot(q_B_values/Delta_S, superfluid_density_yy, "o")
# axs[0].plot(q_B_values/Delta_S, superfluid_density_yy_0, "o")
# axs[0].plot(q_B_values/Delta_S, superfluid_density_xy, "o")
# axs[0].plot(q_B_values/Delta_S, superfluid_density_yy + superfluid_density_xy, "o")
# axs[0].plot(B_values/Delta_S, superfluid_density_yy_0, "o")
axs[0].legend(prop={'size': 4})
axs[0].set_xlabel(r"$q_B/\Delta$")
axs[0].set_ylabel(r"$D_s$")
axs[0].set_title(r"$\lambda/\Delta_S=$" + f"{Lambda/Delta_S}")

axs[1].plot(q_B_values/np.pi, q_eq/np.pi, "o")

axs[1].set_xlabel(r"$q_B/\pi")
axs[1].set_ylabel(r"$q_{eq}/\pi$")


