#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:27:29 2025

@author: gabriel
"""

import numpy as np
import scipy
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y

def get_Hamiltonian(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda):
    """Return the Hamiltonian for a given k."""
    chi_k_plus = gamma * ( (k_x + phi_x)**2 + k_y**2) - mu
    chi_k_minus = gamma * ( (-k_x + phi_x)**2 + k_y**2 ) - mu
    return 1/2 * ( chi_k_plus * np.kron( ( tau_0 + tau_z )/2, sigma_0)
                   - chi_k_minus * np.kron( ( tau_0 - tau_z )/2, sigma_0)
                   - B * np.kron(tau_0, sigma_y)
                   - Delta * np.kron(tau_x, sigma_0)
                   + Lambda * (k_x + phi_x) * np.kron( ( tau_0 + tau_z )/2, sigma_y )
                   + Lambda * (-k_x + phi_x) * np.kron( ( tau_0 - tau_z )/2, sigma_y )
                   - Lambda * k_y * np.kron( tau_z, sigma_x )
                 )

def get_Energies(k_x_values, k_y_values, mu, B, Delta, phi_x, gamma, Lambda):
    """Return the energies of the Hamiltonian at a given k."""
    E = np.zeros((len(k_x_values), len(k_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            H = get_Hamiltonian(k_x, k_y, mu, B, Delta, phi_x, gamma, Lambda)
            E[i, j, :] = np.linalg.eigvalsh(H)
    return E

def get_Hamiltonian_in_polars(k, theta, B_x, B_y, q_B_x, q_B_y, mu, Delta, gamma, Lambda,
                              q_x, q_y, phi_x, phi_y):
    """Return the Hamiltonian for a given k."""
    k_x = k * np.cos(theta)
    k_y = k * np.sin(theta)
    chi_k_plus = gamma * ( (k_x - q_B_x + q_x + phi_x)**2 + (k_y - q_B_y + q_y + phi_y)**2) - mu
    chi_k_minus = gamma * ( (-k_x + phi_x)**2 + (-k_y + phi_y)**2 ) - mu
    return 1/2 * ( chi_k_plus * np.kron( ( tau_0 + tau_z )/2, sigma_0)
                   - chi_k_minus * np.kron( ( tau_0 - tau_z )/2, sigma_0)
                   - B_y * np.kron(tau_0, sigma_y)
                   - B_x * np.kron(tau_0, sigma_x)
                   - Delta * np.kron(tau_x, sigma_0)
                   + Lambda * (k_x + phi_x) * np.kron( ( tau_0 + tau_z )/2, sigma_y )
                   + Lambda * (-k_x + phi_x) * np.kron( ( tau_0 - tau_z )/2, sigma_y )
                   - Lambda * (k_y + phi_y) * np.kron( ( tau_0 + tau_z )/2, sigma_x )
                   - Lambda * (-k_y + phi_y) * np.kron( ( tau_0 - tau_z )/2, sigma_x )
                 )

def get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    """Return the energies of the Hamiltonian at a given k."""
    E = np.zeros((len(k_values), len(theta_values), 4))
    for i, k in enumerate(k_values):
        for j, theta in enumerate(theta_values):
            H = get_Hamiltonian_in_polars(k, theta, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
            E[i, j, :] = np.linalg.eigvalsh(H)
    return E

def get_Analytic_energies_at_k_y_zero(k_x, mu, B, Delta, phi_x, gamma, Lambda):
    return np.array([
        1/2 * (
            (B - Lambda*phi_x) + 2*k_x*gamma*phi_x + np.sqrt(Delta**2 + (k_x**2 * gamma - Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            (B - Lambda*phi_x) + 2*k_x*gamma*phi_x - np.sqrt(Delta**2 + (k_x**2 * gamma - Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            -(B - Lambda*phi_x) + 2*k_x*gamma*phi_x + np.sqrt(Delta**2 + (k_x**2 * gamma + Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        1/2 * (
            -(B - Lambda*phi_x) + 2*k_x*gamma*phi_x - np.sqrt(Delta**2 + (k_x**2 * gamma + Lambda*k_x - mu + gamma*phi_x**2)**2)
        ),
        ])
