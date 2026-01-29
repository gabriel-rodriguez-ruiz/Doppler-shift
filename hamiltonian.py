# -*- coding: utf-8 -*-

import numpy as np
from pauli_matrices import (tau_0, tau_y, sigma_0, tau_z, sigma_x, sigma_y, tau_x)
import scipy

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1, q_B_x, q_B_y, q_x, q_y):
    r""" A semiconductor plane over a superconductor plane. The semiconductor
    has spin-orbit coupling and magnetic field.
    
    .. math::
        H_\mathbf{k} = \frac{1}{2} (H_s + H_S + H_{w_1})
        
        H_s = -2w_s\left(\cos(k_x)\cos(\phi_x + 2 q_{B_x} + q_x) + \cos(k_y)\cos(\phi_y + 2 q_{B_y} + q_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\phi_x + 2 q_{B_x} + q_x) + \sin(k_y)\sin(\phi_y + 2 q_{B_y} + q_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + 2\lambda\left(\sin(k_x)\cos(\phi_x + 2 q_{B_x} + q_x)\tau_z\sigma_y
        + \cos(k_x)\sin(\phi_x + 2 q_{B_x} + q_x)\tau_0\sigma_y
        - \sin(k_y)\cos(\phi_y + 2 q_{B_y} + q_y)\tau_z\sigma_x
        - \sin(k_y)\sin(\phi_y + 2 q_{B_y} + q_y)\tau_0\sigma_x
        - B_x\tau_0\sigma_x - B_y\tau_0\sigma_y \right)
        
        H_S = -2w_S\left(\cos(k_x)\cos(\phi_x + q_x) + \cos(k_y)\cos(\phi_y + q_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\phi_x + q_x) + \sin(k_y)\sin(\phi_y + q_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + \Delta \tau_x\sigma_0
        
        H_{w_1} = -w_1 \alpha_x\tau_z\sigma_0
            
    """
    H_s = (
        -2*w_s*((np.cos(k_x)*np.cos(phi_x+2*q_B_x+q_x) + np.cos(k_y)*np.cos(phi_y+2*q_B_y+q_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x+2*q_B_x+q_x) + np.sin(k_y)*np.sin(phi_y+2*q_B_y+q_y))
               * np.kron(tau_0, sigma_0)) - mu_s * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x+2*q_B_x+q_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x+2*q_B_x+q_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y+2*q_B_y+q_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y+2*q_B_y+q_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
         + Delta_s*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_S = (
        -2*w_S*((np.cos(k_x)*np.cos(phi_x+q_x) + np.cos(k_y)*np.cos(phi_y+q_y))   #without q in the superconductor
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x+q_x) + np.sin(k_y)*np.sin(phi_y+q_y))      # added minus sign because of flux plaquette
               * np.kron(tau_0, sigma_0)) - mu_S * np.kron(tau_z, sigma_0)
        - B_x_S*np.kron(tau_0, sigma_x) - B_y_S*np.kron(tau_0, sigma_y)
        + Delta_S*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_w_1 = 1/2 * ( -w_1 * np.kron(tau_z, sigma_0) )
    H = np.block([
            [H_s, H_w_1],
            [H_w_1.conj().T, H_S]
        ])
    return H