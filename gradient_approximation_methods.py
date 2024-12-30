# -*- coding: utf-8 -*-
"""
Gradient Approximation Methods

This module implements various methods for estimating the gradient of a function. These methods are used in optimization and derivative-free settings,
particularly in scenarios where direct computation of gradients is costly or infeasible. 

The methods included are designed to approximate gradients under different conditions, such as noise-free and noisy data, using deterministic
or stochastic approaches. Each method has been adapted to align with the theoretical framework presented in the "Normalized Mixed Finite Differences
(NMXFD)" scheme, as described in the published work:

A Mixed Finite Differences Scheme for Gradient Approximation
Journal of Optimization Theory and Applications (2022), Vol. 194, Issue 1, Pages 1-24
Marco Boresta et al.
DOI: 10.1007/s10957-021-01994-w

Key Features:
- Implementation of the NMXFD scheme for enhanced gradient estimation accuracy.
- Centralized definitions of standard finite difference methods for comparison.
- Modular design for integration with broader optimization workflows.

Each function includes detailed comments to facilitate understanding and future maintenance.

"""
import numpy as np
def gaussian_derivative(x, mu, sigma):
    """
    Compute the derivative of a Gaussian function with respect to x.

    Args:
        x (float or np.ndarray): Input variable(s) where the derivative is computed.
        mu (float): Mean (center) of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        float or np.ndarray: Value of the Gaussian derivative at the given x.
    """
    # Compute the denominator (normalization factor for Gaussian derivative)
    denominator = np.sqrt(2 * np.pi) * (sigma**3)
    # Compute the numerator (scaled Gaussian function)
    numerator = (x - mu) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    # Return the derivative
    return -numerator / denominator


def cGSG(x, function, num_directions, sigma, lambda_estimate, seed=8):
    """
    Estimate the gradient of a function using Gaussian smoothing with a specified number of directions.

    Args:
        x (np.ndarray): Point at which the gradient is estimated.
        function (callable): Objective function to evaluate.
        num_directions (int): Number of random directions to sample for the estimate.
        sigma (float): Perturbation scale (smoothing parameter).
        lambda_estimate (float): Parameter passed to the function to manage noise or scaling.
        seed (int, optional): Seed for reproducibility of random directions. Defaults to 8.

    Returns:
        np.ndarray: Estimated gradient at point x using Gaussian smoothing.
    """
    np.random.seed(seed)  # Set the seed for reproducibility

    gradient_estimate = np.zeros_like(x)  # Initialize gradient estimate

    for _ in range(num_directions):
        # Generate a random direction sampled from a standard normal distribution
        direction = np.random.normal(0, 1, x.shape[0])

        # Compute finite difference approximation for the given direction
        positive_eval = function(x + sigma * direction, lambda_estimate)
        negative_eval = function(x - sigma * direction, lambda_estimate)
        directional_estimate = (positive_eval - negative_eval) * direction

        # Accumulate the contribution of the current direction
        gradient_estimate += directional_estimate

    # Normalize the gradient estimate by the total number of samples and the scaling factor
    gradient_estimate /= (2 * num_directions * sigma)

    return gradient_estimate

def NMXFD(x, function, num_points, sigma, lambda_estimate, range_integral, seed=8):
    """
    Estimate the gradient of a function using the Normalized Mixed Finite Differences (NMXFD) scheme.

    Args:
        x (np.ndarray): Input point where the gradient is estimated.
        function (callable): Objective function to evaluate.
        num_points (int): Number of evaluation points in the integration range.
        sigma (float): Smoothing parameter (step size for finite differences).
        lambda_estimate (float): Parameter for managing noise or scaling.
        range_integral (tuple): Range of integration (start, end) for finite differences.
        seed (int, optional): Random seed for reproducibility. Defaults to 8.

    Returns:
        np.ndarray: Estimated gradient vector at point x.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    s = x.shape[0]  # Dimension of the input
    gradient = np.zeros(s, dtype='d')  # Initialize gradient vector

    # Define integration points (u) and step size (delta)
    u = np.linspace(range_integral[0], range_integral[1], num_points, dtype='d')
    delta = u[1] - u[0]

    # Compute derivative of the Gaussian kernel
    phi_derivative = -gaussian_derivative(u, 0, 1).reshape(-1, 1)

    for j in range(s):
        # Create perturbed copies of x for integration
        perturbed_x = np.copy(x).reshape(-1, 1)
        results = []

        for i in range(num_points):
            perturbed_x_copy = np.copy(perturbed_x)
            perturbed_x_copy[j] = x[j] + sigma * u[i]
            results.append(function(perturbed_x_copy, lambda_estimate))

        results = np.array(results).reshape(-1, 1)

        # Compute finite differences and corresponding coefficients
        differences = np.array([
            results[::-1][k] - results[k] for k in range(num_points // 2)
        ]).reshape(-1, 1)

        phi_coefficients = np.abs(phi_derivative[:num_points // 2])

        # Calculate integration coefficients
        h = (range_integral[1] - range_integral[0]) / (num_points - 1)
        mult = u[::-1][:num_points // 2].reshape(-1, 1).astype(float)
        coefficients = phi_coefficients * mult * h
        coefficients[0] /= 2  # Adjust for boundary conditions

        # Normalize coefficients if needed
        normalized_coefficients = coefficients / np.sum(coefficients)

        # Compute gradient contribution for the current dimension
        output = normalized_coefficients * differences / (mult * sigma) / 2
        gradient[j] = np.sum(output)

    return gradient


def ACFD(x, function, num_points, sigma, lambda_estimate, range_integral, seed=8):
    """
    Compute the Average Central Finite Differences (ACFD) for gradient estimation.

    Args:
        x (np.ndarray): Input point where the gradient is estimated.
        function (callable): Objective function to evaluate.
        num_points (int): Number of evaluation points in the integration range.
        sigma (float): Smoothing parameter (step size for finite differences).
        lambda_estimate (float): Parameter for managing noise or scaling.
        range_integral (tuple): Range of integration (start, end) for finite differences.
        seed (int, optional): Random seed for reproducibility. Defaults to 8.

    Returns:
        np.ndarray: Estimated gradient vector at point x using ACFD.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    s = x.shape[0]  # Dimension of the input
    gradient = np.zeros(s, dtype='d')  # Initialize gradient vector

    # Define integration points (u) and step size (delta)
    u = np.linspace(range_integral[0], range_integral[1], num_points, dtype='d')
    delta = u[1] - u[0]  # Step size between integration points

    # Compute derivative of the Gaussian kernel
    phi_derivative = -gaussian_derivative(u, 0, 1).reshape(-1, 1)

    for j in range(s):
        # Create perturbed copies of x for integration
        perturbed_x = np.copy(x).reshape(-1, 1)
        results = []

        for i in range(num_points):
            perturbed_x_copy = np.copy(perturbed_x)
            perturbed_x_copy[j] = x[j] + sigma * u[i]
            results.append(function(perturbed_x_copy, lambda_estimate))

        results = np.array(results).reshape(-1, 1)

        # Compute finite differences
        differences = np.array([
            results[::-1][k] - results[k] for k in range(num_points // 2)
        ]).reshape(-1, 1)

        # Calculate central finite differences
        mult = u[::-1][:num_points // 2].reshape(-1, 1).astype(float)
        central_differences = differences / (2 * mult * sigma)

        # Average the central finite differences for the current dimension
        gradient[j] = np.mean(central_differences)

    return gradient


def FFD(x, f, sigma, lambda_estimate, seed=8):
    """
    Compute the Forward Finite Differences (FFD) for gradient estimation.

    Args:
        x (np.ndarray): Input point where the gradient is estimated.
        f (callable): Objective function to evaluate.
        sigma (float): Step size for finite differences.
        lambda_estimate (float): Parameter for managing noise or scaling.
        seed (int, optional): Random seed for reproducibility. Defaults to 8.

    Returns:
        np.ndarray: Estimated gradient vector at point x using FFD.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    s = x.shape[0]  # Dimension of the input
    gradient = np.zeros(s)  # Initialize gradient vector

    # Evaluate the function at the original point
    fx = f(x, lambda_estimate)

    for i in range(s):
        # Perturb the current dimension
        x_temp = np.copy(x)
        x_temp[i] += sigma

        # Compute the forward finite difference for the current dimension
        gradient[i] = (f(x_temp, lambda_estimate) - fx) / sigma

    return gradient



def CFD(x, f, sigma, lambda_estimate, seed=8):
    """
    Compute the Central Finite Differences (CFD) for gradient estimation.

    Args:
        x (np.ndarray): Input point where the gradient is estimated.
        f (callable): Objective function to evaluate.
        sigma (float): Step size for finite differences.
        lambda_estimate (float): Parameter for managing noise or scaling.
        seed (int, optional): Random seed for reproducibility. Defaults to 8.

    Returns:
        np.ndarray: Estimated gradient vector at point x using CFD.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    s = x.shape[0]  # Dimension of the input
    gradient = np.zeros(s, dtype=np.float64)  # Initialize gradient vector

    for i in range(s):
        # Perturb the current dimension in positive and negative directions
        x_temp1 = np.copy(x)
        x_temp2 = np.copy(x)
        x_temp1[i] += sigma
        x_temp2[i] -= sigma

        # Compute the central finite difference for the current dimension
        gradient[i] = (f(x_temp1, lambda_estimate) - f(x_temp2, lambda_estimate)) / (2 * sigma)

    return gradient


def GSG(x, function, num_samples, sigma, lambda_estimate, seed):
    """
    Compute the Gaussian Smoothed Gradient (GSG) for gradient estimation.

    Args:
        x (np.ndarray): Input point where the gradient is estimated.
        function (callable): Objective function to evaluate.
        num_samples (int): Number of random directions for smoothing.
        sigma (float): Smoothing parameter (scale of perturbations).
        lambda_estimate (float): Parameter for managing noise or scaling.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Estimated gradient vector at point x using GSG.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    estimate = np.zeros_like(x)  # Initialize gradient estimate

    # Evaluate the function at the original point
    fx = function(x, lambda_estimate)

    for _ in range(num_samples):
        # Generate a random direction from the standard normal distribution
        direction = np.random.normal(0, 1, x.shape[0])

        # Update the estimate with the contribution from the current direction
        estimate += (function(x + sigma * direction, lambda_estimate) - fx) * direction

    # Normalize the estimate by the number of samples and smoothing parameter
    estimate /= (sigma * num_samples)

    return estimate