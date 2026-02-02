#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:58:28 2025

@author: gabriel
"""

from diagonalization import get_Energies_in_polars, get_Analytic_energies_at_k_y_zero
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.path import Path

def find_nonzero_domains_numpy(arr):
    """
    NumPy version for finding non-zero domains in each row.
    """
    result = []
    
    for i in range(arr.shape[0]):
        row = arr[i]
        # Find where non-zero values start and end
        nonzero_mask = row != 0
        if not np.any(nonzero_mask):
            result.append([])
            continue
            
        # Find boundaries where values change from 0 to non-zero or vice versa
        boundaries = np.where(nonzero_mask[1:] != nonzero_mask[:-1])[0] + 1
        
        # If first element is non-zero, add start at 0
        if nonzero_mask[0]:
            starts = [0] + boundaries[1::2].tolist()
            ends = boundaries[::2].tolist()
        else:
            starts = boundaries[::2].tolist()
            ends = boundaries[1::2].tolist()
        
        # If last element is non-zero, add end at last index
        if nonzero_mask[-1]:
            ends.append(len(row) - 1)
        
        row_domains = list(zip(starts, ends))
        result.append(row_domains)
    
    return result

def find_all_roots(f, x_range=(-10, 10), num_points=10000, tol=1e-8):
    """
    Find all real roots of a function f(x) in a given range.
    
    Parameters:
    f: function to find roots of
    x_range: tuple (min, max) search range
    num_points: number of points for initial sampling
    tol: tolerance for root uniqueness
    """
    # Sample the function to find sign changes
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = f(x_vals)
    
    roots = []
    
    # Find intervals where sign changes (potential roots)
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i + 1] <= 0:  # Sign change or zero
            # Refine the root using brentq
            try:
                root_val = scipy.optimize.brentq(f, x_vals[i], x_vals[i + 1])
                # Check if this root is distinct from previously found ones
                if not any(abs(root_val - r) < tol for r in roots):
                    roots.append(root_val)
            except (ValueError, RuntimeError):
                continue
    
    return np.array(roots)

def get_roots_at_k_y_zero(mu, B, Delta, phi_x, gamma, Lambda, k_F):
    """
    Returns an ndarray with all the roots at k_y=0.
    """
    f = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[2]
    g = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[1]
    h = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[0]
    l = lambda x: get_Analytic_energies_at_k_y_zero(x, mu, B, Delta, phi_x, gamma, Lambda)[3]
    roots = find_all_roots(f, x_range=(-1.01*k_F, -0.99*k_F))
    roots = np.append(roots, find_all_roots(g, x_range=(-1.01*k_F, -0.99*k_F)))
    roots = np.append(roots, find_all_roots(h, x_range=(-1.01*k_F, -0.99*k_F)))
    roots = np.append(roots, find_all_roots(l, x_range=(-1.01*k_F, -0.99*k_F)))
    roots = np.append(roots, find_all_roots(f, x_range=(0.99*k_F, 1.01*k_F)))
    roots = np.append(roots, find_all_roots(g, x_range=(0.99*k_F, 1.01*k_F)))
    roots = np.append(roots, find_all_roots(h, x_range=(0.99*k_F, 1.01*k_F)))
    roots = np.append(roots, find_all_roots(l, x_range=(0.99*k_F, 1.01*k_F)))
    return roots


def get_pockets_contour(N, mu, B, Delta, phi_x, gamma, Lambda, k_F):
    """
    Returns four interpolation for the pockets of each energy.
    """
    roots = get_roots_at_k_y_zero(mu, B, Delta, phi_x, gamma, Lambda, k_F)
    minima_k_value = np.min(abs(roots))
    maxima_k_value = np.max(abs(roots))
    #radius_values = np.linspace(0.99*minima_k_value, 1.01*maxima_k_value, N)
    radius_values = np.linspace(0.99, 1.01, N)*k_F
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
    radius, theta = np.meshgrid(radius_values, theta_values)
    Energies_polar = get_Energies_in_polars(radius_values, theta_values, mu, B, Delta, phi_x, gamma, Lambda)
    contours = []
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(4):
        contour = ax.contour(theta, radius, Energies_polar[:,:, i].T, levels=[0.0], colors=f"C{i}")
        contours.append(contour)
    plt.close(fig)  # Close the plot, we only need the data
    #plt.show()
    interpolation = []
    for j in range(4):
        CS = contours[j]
        # Extract the segments and create paths
        paths_by_level = {}
        for i, level_segs in enumerate(CS.allsegs):
            level = CS.levels[i]
            if level not in paths_by_level:
                paths_by_level[level] = []
            
            # Check if the segment is a closed loop
            for seg in level_segs:
                if len(seg)==0:
                    break
                else:
                    is_closed = np.all(seg[0] == seg[-1])
                    # Only use closed paths for point-in-polygon tests
                    if is_closed and len(seg) > 2:
                        # Interpolate the segment to smooth the path
                        distance = np.cumsum(np.sqrt(np.diff(seg[:,0])**2 + np.diff(seg[:,1])**2))
                        distance = np.insert(distance, 0, 0)
                        
                        num_interp_points = 200
                        new_distances = np.linspace(0, distance[-1], num_interp_points)
                        interp_x = scipy.interpolate.interp1d(distance, seg[:, 0], kind='cubic')
                        interp_y = scipy.interpolate.interp1d(distance, seg[:, 1], kind='cubic')
                        
                        interp_seg = np.vstack((interp_x(new_distances), interp_y(new_distances))).T
                        
                        # Create a path from the interpolated segment
                        paths_by_level[level].append(Path(interp_seg))
        interpolation.append(paths_by_level)
    return interpolation, Energies_polar

def is_inside_contour(points, contour_level, paths_by_level_dict):
    """
    Checks if a list of (x,y) points are inside the specified contour level.

    Args:
        points (list or np.ndarray): A list of (x, y) coordinates.
        contour_level (float): The contour level to check against.
        paths_by_level_dict (dict): The dictionary of Path objects.

    Returns:
        np.ndarray: A boolean mask indicating if each point is inside.
    """
    # Check if the requested level exists
    if contour_level not in paths_by_level_dict:
        raise ValueError(f"Contour level {contour_level} not found.")

    all_paths_for_level = paths_by_level_dict[contour_level]
    
    if not all_paths_for_level:
        return np.zeros(len(points), dtype=bool)

    points = np.asarray(points)
    
    # The final mask is the OR of all individual path checks
    is_inside_mask = np.zeros(len(points), dtype=bool)
    for path in all_paths_for_level:
        is_inside_mask |= path.contains_points(points)
        
    return is_inside_mask

def plot_interpolated_contours(N, mu, B, Delta, phi_x, gamma, Lambda, k_F):
    pockets_dictionary, Energies = get_pockets_contour(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
    # Generate a grid of points to test
    test_x = np.linspace(-np.pi/2, 3*np.pi/2, 50)
    test_y = np.linspace(0.99, 1.005, 50) * k_F
    TestX, TestY = np.meshgrid(test_x, test_y)
    test_points = np.vstack((TestX.ravel(), TestY.ravel())).T
    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': 'polar'})
    axs = axs.flatten()
    # Get the mask for the 0 contour level
    for i in range(4):
        paths_by_level = pockets_dictionary[i]
        contour_level_to_test = 0
        is_inside = is_inside_contour(test_points, contour_level_to_test, paths_by_level)
        
        # Create a color mask for plotting
        color_mask = np.where(is_inside, 'red', 'blue')
        
        # Plot the results
        #axs[i].set_title(f'Point in Contour Test (Level {contour_level_to_test})')
        #axs[i].set_xlabel('x')
        #axs[i].set_ylabel('y')
        
        # Draw the interpolated contour
        for path in paths_by_level[contour_level_to_test]:
            axs[i].plot(path.vertices[:, 0], path.vertices[:, 1], 'g-', linewidth=2)
        
        # Plot the test points
        axs[i].scatter(test_points[:, 0], test_points[:, 1], c=color_mask, s=5, alpha=0.5)
        axs[i].set_ylim(0.99*k_F, 1.005*k_F) 
    plt.show()
    
def integrate_pocket(N, mu, B, Delta, phi_x, gamma, Lambda, k_F):
    integral = np.zeros(4)
    pockets_dictionary, Energies = get_pockets_contour(N, mu, B, Delta, phi_x, gamma, Lambda, k_F)
    contour_level_to_test = 0
    roots = get_roots_at_k_y_zero(mu, B, Delta, phi_x, gamma, Lambda, k_F)
    minima_k_value = np.min(abs(roots))
    maxima_k_value = np.max(abs(roots))
    #radius_values = np.linspace(0.99*minima_k_value, 1.01*maxima_k_value, N)
    radius_values = np.linspace(0.99, 1.01, N)*k_F
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
    radius, theta = np.meshgrid(radius_values, theta_values)
    Z = np.zeros((len(radius_values), len(theta_values)))
    for i, pocket in enumerate(pockets_dictionary):
        if len(pocket[0]) == 0:
            # No pocket
            integrand_polar = lambda theta, r: get_Energies_in_polars([r], [theta], mu, B, Delta, phi_x, gamma, Lambda)[0][0][i]
            for j, r in enumerate(radius_values):
                for k, theta in enumerate(theta_values):
                    Z[j, k] = integrand_polar(theta, r)
        else:
            integrand_polar = lambda theta, r: get_Energies_in_polars([r], [theta], mu, B, Delta, phi_x, gamma, Lambda)[0][0][i] * (i%2 + (-1)**i * is_inside_contour([[theta, r]],
                                                                                                                                                     contour_level_to_test, pocket))
    
            Z = np.zeros((len(radius_values), len(theta_values)))
            for j, r in enumerate(radius_values):
                for k, theta in enumerate(theta_values):
                    Z[j, k] = integrand_polar(theta, r)[0]
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, radius_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, theta_values, axis=0)
        
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, theta_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, radius_values, axis=0)
        integral[i] = double_integral
    return integral

def integrate_brute_force(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y):
    integral = np.zeros(4)
    #roots = get_roots_at_k_y_zero(mu, B, Delta, phi_x, gamma, Lambda, k_F)
    #minima_k_value = np.min(abs(roots))
    #maxima_k_value = np.max(abs(roots))
    #radius_values = np.linspace(0.99*minima_k_value, 1.01*maxima_k_value, N)
    
    radius_values = np.linspace(0.99, 1.01, N)*k_F
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
    radius, theta = np.meshgrid(radius_values, theta_values)
    Z = np.zeros((len(radius_values), len(theta_values)))
    for i in range(4):
        for j, r in enumerate(radius_values):
            for k, theta in enumerate(theta_values):
                E = r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
                if E <= 0:
                    Z[j, k] = E
                else:
                    Z[j, k] = 0
        
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, theta_values, axis=0,
                                                   dx=np.diff(theta_values)[0])
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, radius_values,
                                                    axis=0,
                                                    dx=np.diff(radius_values)[0])
        integral[i] = double_integral
    
    low_integral = np.zeros(4)
    """
    radius_values = np.linspace(0, 0.99, N)*k_F
    #radius_values = np.linspace(0, 0.99*minima_k_value, N)
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
    radius, theta = np.meshgrid(radius_values, theta_values)
    Z = np.zeros((len(radius_values), len(theta_values)))
    for i in range(4):
        for j, r in enumerate(radius_values):
            for k, theta in enumerate(theta_values):
                E = get_Energies_in_polars([r], [theta], mu, B, Delta, phi_x, gamma, Lambda)[0][0][i]
                if E <= 0:
                    Z[j, k] = E
                else:
                    Z[j, k] = 0
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, radius_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, theta_values, axis=0)
        
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, theta_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, radius_values, axis=0)
        low_integral[i] = double_integral
    """
    for i in range(2):
        f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
        low_integral[i], abserr = scipy.integrate.dblquad(f, 0, 2*np.pi, 0, 0.99*k_F) 

    
    high_integral = np.zeros(4)
    """
    radius_values = np.linspace(1.01, cut_off/k_F, N)*k_F
    #radius_values = np.linspace(1.01*maxima_k_value, cut_off, N)
    theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
    radius, theta = np.meshgrid(radius_values, theta_values)
    
    Z = np.zeros((len(radius_values), len(theta_values)))
    for i in range(4):
        for j, r in enumerate(radius_values):
            for k, theta in enumerate(theta_values):
                E = get_Energies_in_polars([r], [theta], mu, B, Delta, phi_x, gamma, Lambda)[0][0][i]
                if E <= 0:
                    Z[j, k] = E
                else:
                    Z[j, k] = 0
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, radius_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, theta_values, axis=0)
        
        # Integrate with respect to y first
        inner_integral = scipy.integrate.trapezoid(Z, theta_values, axis=0)
        
        # Integrate the result with respect to x
        double_integral = scipy.integrate.trapezoid(inner_integral, radius_values, axis=0)
        high_integral[i] = double_integral
    """
    for i in range(2):
        f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
        high_integral[i], abserr = scipy.integrate.dblquad(f, 0, 2*np.pi, 1.01*k_F, cut_off) 

    return integral, low_integral, high_integral

def rombND(z, steps=1):
    """
    Romberg ND-integration using samples of a ND function. 
    
    See scipy.integrate.romb for details.
    
    >>> nx, ny, nz = 2**3 + 1, 2**4 + 1, 2**2 + 1
    >>> xlims, ylims, zlims = (0, 1), (0, 2), (0, 1/2)
    >>> z, y, x = np.ogrid[zlims[0]:zlims[1]:nz*1j, ylims[0]:ylims[1]:ny*1j, xlims[0]:xlims[1]:nx*1j]
    >>> dz, dy, dx = (z[-1, 0, 0] - z[0, 0, 0]) / (nz - 1), (y[0, -1, 0] - y[0, 0, 0]) / (ny - 1), (x[0, 0, -1] - x[0, 0, 0]) / (nx - 1)
    >>> integrand = (2 * x + y + z / 2)**2  # int_{x=0}^{1} int_{y=0}^{2} int_{z=0}^{1/2} = 83/16
    >>> np.isclose(rombND(integrand, (dx, dy, dz)), 83/16)
    True
    """
    
    steps = np.resize(steps, (z.ndim,))  # Make it a 1D vector
    for axis in range(z.ndim):
        step = steps[axis]
        if axis == 0:
            integral = [ scipy.integrate.romb(zz, step) for zz in z ]
        else:
            integral = scipy.integrate.romb(integral, step)
            
    return integral

def integrate_Romberg(N, mu, B_x, B_y, q_B_x, q_B_y, Delta, gamma, Lambda, k_F, cut_off, q_x, q_y, phi_x, phi_y):
    q_B = np.sqrt(q_B_x**2 + q_B_y**2)
    q = np.sqrt(q_x**2 + q_y**2)
    k = int(np.log2(N-1))
    N = 2**k + 1
    integral = np.zeros(4)
    
    roots = [0.99*k_F, 1.012*k_F]
    
    if len(roots)==0:   # no pockets
        r_min = 0.99*k_F
        r_max = 1.012*k_F
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, q_B_y, Delta, q_x, gamma, Lambda, q_B_x, q_y)[0][0][i]
            integral[i], c = scipy.integrate.dblquad(f, 0, 2*np.pi, 0, cut_off) 
        low_integral = 0
        high_integral = 0
        
    elif len(roots)==2:   # one pair of pockets
        r_min = min(np.abs(roots))
        r_max = max(np.abs(roots))
        
        radius_values = np.linspace(r_min, r_max, N)
        theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
        radius, theta = np.meshgrid(radius_values, theta_values)
        Z = np.zeros((len(radius_values), len(theta_values)))
        for i in range(4):
            for j, r in enumerate(radius_values):
                for k, theta in enumerate(theta_values):
                    E = r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
                    if E <= 0:
                        Z[j, k] = E
                    else:
                        Z[j, k] = 0
            integral[i] = rombND(Z, steps=(np.diff(theta_values)[0], np.diff(radius_values)[0], ))
        
        low_integral = np.zeros(4)
        
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
            low_integral[i], c = scipy.integrate.dblquad(f, 0, 2*np.pi, 0, r_min) 
    
        
        high_integral = np.zeros(4)
        
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
            high_integral[i], abserr = scipy.integrate.dblquad(f, 0, 2*np.pi, r_max, cut_off) 

    else:   # two pairs of pockets
        roots = np.sort(np.abs(roots))
        r_min_1 = 0.99 * np.abs(roots[0])     
        r_max_1 = 1.01 * np.abs(roots[1])  
        r_min_2 = 0.99 * np.abs(roots[2])
        r_max_2 = 1.01 * np.abs(roots[3])  
        
        radius_values = np.linspace(r_min_1, r_max_1, N)
        theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
        radius, theta = np.meshgrid(radius_values, theta_values)
        Z = np.zeros((len(radius_values), len(theta_values)))
        for i in range(4):
            for j, r in enumerate(radius_values):
                for k, theta in enumerate(theta_values):
                    E = r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
                    if E <= 0:
                        Z[j, k] = E
                    else:
                        Z[j, k] = 0
            integral[i] = rombND(Z, steps=(np.diff(theta_values)[0], np.diff(radius_values)[0], ))
            
        radius_values = np.linspace(r_min_2, r_max_2, N)
        theta_values = np.linspace(-np.pi/2, 3*np.pi/2, N)
        radius, theta = np.meshgrid(radius_values, theta_values)
        Z = np.zeros((len(radius_values), len(theta_values)))
        for i in range(4):
            for j, r in enumerate(radius_values):
                for k, theta in enumerate(theta_values):
                    E = r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
                    if E <= 0:
                        Z[j, k] = E
                    else:
                        Z[j, k] = 0
            integral[i] += rombND(Z, steps=(np.diff(theta_values)[0], np.diff(radius_values)[0], ))
            
        low_integral = np.zeros(4)
        
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
            low_integral[i], c = scipy.integrate.dblquad(f, 0, 2*np.pi, 0, r_min_1)
        
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
            result, c = scipy.integrate.dblquad(f, 0, 2*np.pi, r_max_1, r_min_2) 
            low_integral[i] += result
    
        high_integral = np.zeros(4)
        
        for i in range(2):
            f = lambda r, theta: r * get_Energies_in_polars([r], [theta], mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[0][0][i]
            high_integral[i], abserr = scipy.integrate.dblquad(f, 0, 2*np.pi, r_max_2, cut_off) 
        
    return integral, low_integral, high_integral