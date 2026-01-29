#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 09:42:54 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Space
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')
import scipy
from pathlib import Path
from functions import get_fundamental_energy

data_folder = Path(r"./Data")
file_to_open = data_folder / "Fundamental_energy_q_B_0.015707963267948967_q_x_in_(-0.0628-0)_mu_S_-38.0_L=300_h=0.0001_B=0.0_Delta=0.2_lambda=0.0_w_s=10_w_S=10_w_1=0.25_points=19.npz"
Data = np.load(file_to_open)
q_x_values = Data["q_x_values"]

plt.rcParams.update({
  "text.usetex": True,
})

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
phi_x_values = [0]
phi_y_values = [0]
q_x_values = [0]
q_y_values = [0]
q_B_x_values = np.array([0.01*np.pi])
q_B_y_values = [0]
h = 1e-4
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_calls = 15
n_initial_points = 5
# Define the skewed Mexican hat function in 1D
def wrapper_function(x):
    def function(q_x):
        energy_q_x = get_fundamental_energy(L_x, L_y, w_s, w_S, mu_s, mu_S, Delta_s, Delta_S, B_x,
                               B_y, B_x_S, B_y_S, Lambda, w_1, phi_x_values=phi_x_values, phi_y_values=phi_y_values,
                               q_B_x_values=q_B_x_values, q_B_y_values=q_B_y_values,
                               q_x_values=[q_x], q_y_values=q_y_values)[0, 0]
        return energy_q_x
    return function(x)

# Define the search space
search_space = [(-0.02 * np.pi, 0.02 * np.pi)]  

#%% Plot fundamental energy


q_x_values = np.linspace(-0.02 * np.pi, 0.02 * np.pi)
energy_q_x = np.zeros_like(q_x_values)

for i, q_x in enumerate(q_x_values):
    print(q_x)
    energy_q_x[i] = wrapper_function(q_x)
    


# Vectorized version for plotting
x_plot = q_x_values/np.pi
y_plot = energy_q_x

# Plot the function to see what we're dealing with
plt.figure(figsize=(12, 5))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Skewed Mexican Hat')
plt.xlabel('x')
plt.ylabel('E_0(x)')
# plt.title('1D Skewed Mexican Hat Function')
plt.grid(True, alpha=0.3)
# plt.legend()
plt.show()

#%%3

# Bayesian Optimization setup
def objective_function(x):
    """Objective function to minimize"""
    return wrapper_function(x[0])

print("Running Bayesian Optimization...")

# Run Bayesian Optimization with only 25 function evaluations!
result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=n_calls,                    # Only 15 expensive evaluations!
    n_initial_points=n_initial_points,           # Start with 5 random points
    random_state=None,
    acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
    noise=0.0,
    initial_point_generator="lhs",
    x0=[[-q_B_x_values[0]], [0.], [q_B_x_values[0]]]
)

print("\n=== RESULTS ===")
print(f"Global minimum found at: x = {result.x[0]:.6f}")
print(f"Function value at minimum: f(x) = {result.fun:.6f}")

#%%

# Find the true global minimum for comparison (we only know this because we can evaluate densely)
true_min_idx = np.argmin(y_plot)
true_min_x = x_plot[true_min_idx]
true_min_val = y_plot[true_min_idx]

print(f"True global minimum: x = {true_min_x:.6f}, f(x) = {true_min_val:.6f}")
print(f"Error in position: {abs(result.x[0] - true_min_x):.6f}")

# Plot the results
plt.figure(figsize=(15, 5))

# Plot 1: Function with optimization points
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Fundamental energy', alpha=0.7)

# Plot all evaluation points
for i, (x_val, y_val) in enumerate(zip([xi[0] for xi in result.x_iters], result.func_vals)):
    color = 'red' if i < 5 else 'green'  # Initial points in red, BO points in green
    marker = 'o' if i < 5 else 's'
    alpha = 0.7 if i < 5 else 1.0
    plt.scatter(x_val/np.pi, y_val, c=color, marker=marker, alpha=alpha, s=50)

# Mark the best point found
plt.scatter(result.x[0]/np.pi, result.fun, c='gold', marker='*', s=200, 
           label=f'Best found: x={result.x[0]/np.pi:.5f}', edgecolors='black')

plt.xlabel(r'$q_x/\pi$')
plt.ylabel('f(x)')
plt.title('Bayesian Optimization Progress\n(Red: Initial, Green: BO, Star: Best)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Convergence plot
plt.subplot(1, 2, 2)
plot_convergence(result)
plt.title('Convergence Plot')

plt.tight_layout()
plt.show()

# Print the evaluation history
print("\n=== EVALUATION HISTORY ===")
print("Iteration |      x      |    f(x)    |   Best So Far")
print("-" * 55)
best_so_far = float('inf')
for i, (x_val, f_val) in enumerate(zip(np.array(result.x_iters)/np.pi, result.func_vals)):
    if f_val < best_so_far:
        best_so_far = f_val
    print(f"{i+1:>8} | {x_val[0]:>10.5f} | {f_val:>9.9f} | {best_so_far:>12.9f}")
    