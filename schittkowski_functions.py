# -*- coding: utf-8 -*-
"""
Implementation of Klaus Schittkowski's Test Problems and Their Derivatives:

@book{schittkowski2012more,
  title={More test examples for nonlinear programming codes},
  author={Schittkowski, Klaus},
  volume={282},
  year={2012},
  publisher={Springer Science \& Business Media}
}

This script provides implementations of various optimization test problems
introduced by Klaus Schittkowski, along with their analytical derivatives.
The problems include quadratic functions, polynomial functions, and functions
with exponential or trigonometric terms, among others. Many of these functions
also incorporate noise to simulate real-world scenarios.



The script contains:
1. Functions representing the objective of the optimization problems.
2. Corresponding derivatives for gradient-based optimization methods.
3. Specific implementations of matrix-based and vectorized computations 
   for efficient performance.

Features:
- Each problem is defined with its own unique function, and its derivative
  is provided as a separate function for ease of gradient-based optimization.
- Functions support the addition of Gaussian noise controlled by a `lambdaError`
  parameter.
- Includes utility functions for calculating second and third derivatives of 
  Gaussian distributions.

Examples:
- `problem201` defines a quadratic function, while `derivative_problem201`
  computes its gradient.
- `problem266` incorporates matrix and vector operations to simulate complex 
  constraints in optimization.

Usage:
- Import this module in optimization scripts requiring test problems for
  benchmarking or gradient estimation.
- Call the required problem and derivative function with the desired inputs.

Author:
- Original Author: Marco Boresta
"""



import numpy as np
import scipy
from scipy.sparse import diags
from scipy.linalg import hilbert


def problem201(x, lambda_noise):
    """
    Compute the objective function for Problem 201 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = 4 * ((x1 - 5) ** 2) + (x2 - 6) ** 2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem201(x):
    """
    Compute the gradient of the objective function for Problem 201.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([8 * (x1 - 5), 2 * (x2 - 6)])
    return gradient

def problem202(x, lambda_noise):
    """
    Compute the objective function for Problem 202 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    term1 = -13 + x1 - 2 * x2 + 5 * (x2 ** 2) - (x2 ** 3)
    term2 = -29 + x1 - 14 * x2 + x2 ** 2 + x2 ** 3
    function_value = term1 ** 2 + term2 ** 2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem202(x):
    """
    Compute the gradient of the objective function for Problem 202.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    term1 = -13 + x1 - 2 * x2 + 5 * (x2 ** 2) - (x2 ** 3)
    term2 = -29 + x1 - 14 * x2 + x2 ** 2 + x2 ** 3
    gradient = np.array([
        2 * term1 + 2 * term2,
        2 * term1 * (-2 + 10 * x2 - 3 * (x2 ** 2)) + 2 * term2 * (-14 + 2 * x2 + 3 * (x2 ** 2))
    ])
    return gradient

def problem203(x, lambda_noise):
    """
    Compute the objective function for Problem 203 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    term1 = 1.5 - x1 * (1 - x2)
    term2 = 2.25 - x1 * (1 - x2 ** 2)
    term3 = 2.625 - x1 * (1 - x2 ** 3)
    function_value = term1 ** 2 + term2 ** 2 + term3 ** 2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem203(x):
    """
    Compute the gradient of the objective function for Problem 203.

    Args:
        x (np.ndarray): Input array of size 2 representing the variables [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    term1 = 1.5 - x1 * (1 - x2)
    term2 = 2.25 - x1 * (1 - x2 ** 2)
    term3 = 2.625 - x1 * (1 - x2 ** 3)
    gradient = np.array([
        -2 * term1 * (1 - x2) - 2 * term2 * (1 - x2 ** 2) - 2 * term3 * (1 - x2 ** 3),
        2 * term1 * x1 + 2 * term2 * 2 * x1 * x2 + 2 * term3 * 3 * x1 * (x2 ** 2)
    ])
    return gradient




def problem204(x, lambda_noise):
    """
    Compute the objective function for Problem 204 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 reshaped to column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.13294, -0.244378, 0.325895]).reshape(-1, 1)
    D = np.array([2.5074, -1.36401, 1.02282]).reshape(-1, 1)
    H = np.array([[-0.564255, 0.392417], [-0.404979, 0.927589], [-0.0735084, 0.535493]])
    B = np.linalg.inv(H.T @ H)

    # Objective function components
    F = A + np.dot(H, x) + 0.5 * ((x.T @ B @ x) * D)
    noise = np.random.normal(0, lambda_noise)
    fun = F.T @ F
    return fun[0][0] + noise


def derivative_problem204(x):
    """
    Compute the gradient of the objective function for Problem 204.

    Args:
        x (np.ndarray): Input array reshaped to column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.13294, -0.244378, 0.325895]).reshape(-1, 1)
    D = np.array([2.5074, -1.36401, 1.02282]).reshape(-1, 1)
    H = np.array([[-0.564255, 0.392417], [-0.404979, 0.927589], [-0.0735084, 0.535493]])
    B = np.linalg.inv(H.T @ H)

    # Compute F
    F = A + np.dot(H, x) + 0.5 * ((x.T @ B @ x) * D)

    # Compute gradient
    gradient = np.zeros_like(x, dtype='d')
    for i in range(3):
        term1 = H[i].reshape(-1, 1)  # Ensure proper shape for H[i]
        term2 = 0.5 * ((x.T @ (B + B.T)) * D[i])  # Match original element-wise multiplication
        gradient += 2 * F[i] * (term1 + term2.reshape(-1, 1))

    return gradient.flatten()

def problem205(x, lambda_noise):
    """
    Compute the objective function for Problem 205 with optional noise.
    This is identical to Problem 203.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    return problem203(x, lambda_noise)

def derivative_problem205(x):
    """
    Compute the gradient of the objective function for Problem 205.
    This is identical to the derivative of Problem 203.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    return derivative_problem203(x)

def problem206(x, lambda_noise):
    """
    Compute the objective function for Problem 206 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (x2 - x1**2)**2 + 100 * (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem206(x):
    """
    Compute the gradient of the objective function for Problem 206.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        2 * (x2 - x1**2) * (-2 * x1) - 200 * (1 - x1),
        2 * (x2 - x1**2)
    ])
    return gradient

def problem207(x, lambda_noise):
    """
    Compute the objective function for Problem 207 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (x2 - x1**2)**2 + (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem207(x):
    """
    Compute the gradient of the objective function for Problem 207.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        2 * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1),
        2 * (x2 - x1**2)
    ])
    return gradient

def problem208(x, lambda_noise):
    """
    Compute the objective function for Problem 208 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem208(x):
    """
    Compute the gradient of the objective function for Problem 208.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        200 * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])
    return gradient

def problem209(x, lambda_noise):
    """
    Compute the objective function for Problem 209 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (10**4) * (x2 - x1**2)**2 + (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem209(x):
    """
    Compute the gradient of the objective function for Problem 209.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        2 * (10**4) * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1),
        2 * (10**4) * (x2 - x1**2)
    ])
    return gradient

def problem210(x, lambda_noise):
    """
    Compute the objective function for Problem 210 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (10**6) * (x2 - x1**2)**2 + (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem210(x):
    """
    Compute the gradient of the objective function for Problem 210.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        2 * (10**6) * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1),
        2 * (10**6) * (x2 - x1**2)
    ])
    return gradient


def problem211(x, lambda_noise):
    """
    Compute the objective function for Problem 211 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = 100 * (x2 - x1**3)**2 + (1 - x1)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem211(x):
    """
    Compute the gradient of the objective function for Problem 211.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    gradient = np.array([
        200 * (x2 - x1**3) * (-3 * x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**3)
    ])
    return gradient

def problem212(x, lambda_noise):
    """
    Compute the objective function for Problem 212 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    term = 4 * (x1 + x2) + (x1 - x2) * ((x1 - 2)**2 + x2**2 - 1)
    function_value = (4 * (x1 + x2))**2 + term**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem212(x):
    """
    Compute the gradient of the objective function for Problem 212.

    Args:
        x (np.ndarray): Input array of size 2 representing [x1, x2].

    Returns:
        np.ndarray: Gradient array [df/dx1, df/dx2].
    """
    x1, x2 = x[0], x[1]
    term = 4 * (x1 + x2) + (x1 - x2) * ((x1 - 2)**2 + x2**2 - 1)
    gradient = np.array([
        32 * (x1 + x2) + 2 * term * (4 + ((x1 - 2)**2 + x2**2 - 1) + (x1 - x2) * 2 * (x1 - 2)),
        32 * (x1 + x2) + 2 * term * (4 - ((x1 - 2)**2 + x2**2 - 1) + (x1 - x2) * 2 * x2)
    ])
    return gradient

def problem213(x, lambda_noise):
    """
    Compute the objective function for Problem 213 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2 = x[0], x[1]
    function_value = (10 * (x1 - x2)**2 + (x1 - 1)**2)**4
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem213(x):
    """
    Compute the gradient of the objective function for Problem 213.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    x1, x2 = x[0], x[1]
    term = 10 * (x1 - x2)**2 + (x1 - 1)**2
    gradient = np.array([
        (4 * term**3) * (20 * (x1 - x2) + 2 * (x1 - 1)),
        (4 * term**3) * (-20 * (x1 - x2))
    ]).flatten()
    return gradient

def problem214(x, lambda_noise):
    """
    Compute the objective function for Problem 214 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2 = x[0], x[1]
    function_value = (10 * (x1 - x2)**2 + (x1 - 1)**2)**0.25
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem214(x):
    """
    Compute the gradient of the objective function for Problem 214.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    x1, x2 = x[0], x[1]
    term = 10 * (x1 - x2)**2 + (x1 - 1)**2
    gradient = np.array([
        (0.25 * term**-0.75) * (20 * (x1 - x2) + 2 * (x1 - 1)),
        (0.25 * term**-0.75) * (-20 * (x1 - x2))
    ]).flatten()
    return gradient

def problem240(x, lambda_noise):
    """
    Compute the objective function for Problem 240 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    function_value = (x1 - x2 + x3)**2 + (-x1 + x2 + x3)**2 + (x1 + x2 - x3)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem240(x):
    """
    Compute the gradient of the objective function for Problem 240.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    gradient = np.array([
        2 * (x1 - x2 + x3) - 2 * (-x1 + x2 + x3) + 2 * (x1 + x2 - x3),
        -2 * (x1 - x2 + x3) + 2 * (-x1 + x2 + x3) + 2 * (x1 + x2 - x3),
        2 * (x1 - x2 + x3) + 2 * (-x1 + x2 + x3) - 2 * (x1 + x2 - x3)
    ]).flatten()
    return gradient



def problem241(x, lambda_noise):
    """
    Compute the objective function for Problem 241 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    
    f1x = x1**2 + x2**2 + x3**2 - 1
    f2x = x1**2 + x2**2 + (x3 - 2)**2 - 1
    f3x = x1 + x2 + x3 - 1
    f4x = x1 + x2 - x3 + 1
    f5x = x1**3 + 3 * (x2**2) + (5 * x3 - x1 + 1)**2 - 36
    
    function_value = f1x**2 + f2x**2 + f3x**2 + f4x**2 + f5x**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem241(x):
    """
    Compute the gradient of the objective function for Problem 241.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]

    f1x = x1**2 + x2**2 + x3**2 - 1
    f2x = x1**2 + x2**2 + (x3 - 2)**2 - 1
    f3x = x1 + x2 + x3 - 1
    f4x = x1 + x2 - x3 + 1
    f5x = x1**3 + 3 * (x2**2) + (5 * x3 - x1 + 1)**2 - 36

    gradient = np.zeros(x.shape[0], dtype='d')
    gradient[0] = 2 * f1x * (2 * x1) + 2 * f2x * (2 * x1) + 2 * f3x + 2 * f4x + 2 * f5x * (3 * (x1**2) - 2 * (5 * x3 - x1 + 1))
    gradient[1] = 2 * f1x * (2 * x2) + 2 * f2x * (2 * x2) + 2 * f3x + 2 * f4x + 2 * f5x * (6 * x2)
    gradient[2] = 2 * f1x * (2 * x3) + 2 * f2x * (2 * (x3 - 2)) + 2 * f3x - 2 * f4x + 2 * f5x * (10 * x3 - 2 * x1 + 2) * 5
    return gradient

def problem243(x, lambda_noise):
    """
    Compute the objective function for Problem 243 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.14272, -0.184918, -0.521869, -0.685306]).reshape(-1, 1)
    D = np.array([1.75168, -1.35195, -0.479048, -0.3648]).reshape(-1, 1)
    B = np.array([[2.95137, 4.87407, -2.0506],
                  [4.87407, 9.39321, -3.93181],
                  [9.39321, -3.93189, 2.64745]])
    G = np.array([[-0.564255, 0.392417, -0.404979],
                  [0.927589, -0.0735084, 0.535493],
                  [-0.658799, -0.636666, -0.681091],
                  [-0.869487, 0.586387, 0.289826]])

    F = A + np.dot(G, x) + 0.5 * ((x.T @ B) @ x) * D
    noise = np.random.normal(0, lambda_noise)
    function_value = F.T @ F
    return function_value[0][0] + noise

def derivative_problem243(x):
    """
    Compute the gradient of the objective function for Problem 243.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.14272, -0.184918, -0.521869, -0.685306]).reshape(-1, 1)
    D = np.array([1.75168, -1.35195, -0.479048, -0.3648]).reshape(-1, 1)
    B = np.array([[2.95137, 4.87407, -2.0506],
                  [4.87407, 9.39321, -3.93181],
                  [9.39321, -3.93189, 2.64745]])
    G = np.array([[-0.564255, 0.392417, -0.404979],
                  [0.927589, -0.0735084, 0.535493],
                  [-0.658799, -0.636666, -0.681091],
                  [-0.869487, 0.586387, 0.289826]])

    F = A + np.dot(G, x) + 0.5 * ((x.T @ B) @ x) * D
    gradient = np.zeros((1, x.shape[0]), dtype='d')
    for i in range(4):
        gradient += 2 * F[i] * (G[i] + 0.5 * ((x.T @ (B + B.T)) * D[i]))
    return gradient.flatten()

def problem244(x, lambda_noise):
    """
    Compute the objective function for Problem 244 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    zvect = 0.1 * np.arange(1, 11)
    yvect = np.exp(-zvect) - 5 * np.exp(-10 * zvect)

    function_value = np.sum((np.exp(-x1 * zvect) - x3 * np.exp(-x2 * zvect) - yvect)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem244(x):
    """
    Compute the gradient of the objective function for Problem 244.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    zvect = 0.1 * np.arange(1, 11)
    yvect = np.exp(-zvect) - 5 * np.exp(-10 * zvect)

    gradient = np.zeros(x.shape[0], dtype='d')
    for z, y in zip(zvect, yvect):
        gradient[0] += 2 * (np.exp(-x1 * z) - x3 * np.exp(-x2 * z) - y) * (-z * np.exp(-x1 * z))
        gradient[1] += 2 * (np.exp(-x1 * z) - x3 * np.exp(-x2 * z) - y) * (x3 * z * np.exp(-x2 * z))
        gradient[2] += 2 * (np.exp(-x1 * z) - x3 * np.exp(-x2 * z) - y) * (-np.exp(-x2 * z))
    return gradient.flatten()
    

def problem245(x, lambda_noise):
    """
    Compute the objective function for Problem 245 with optional noise.

    Args:
        x (np.ndarray): Input array of size 3 reshaped to a column vector [x1, x2, x3].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    jvect = np.arange(1, 11)

    term1 = np.exp(-jvect * x1 / 10) - np.exp(-jvect * x2 / 10)
    term2 = x3 * (np.exp(-jvect / 10) - np.exp(-jvect))
    function_value = np.sum((term1 - term2)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem245(x):
    """
    Compute the gradient of the objective function for Problem 245.

    Args:
        x (np.ndarray): Input array of size 3 reshaped to a column vector [x1, x2, x3].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3].
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    jvect = np.arange(1, 11)
    
    term1 = np.exp(-jvect * x1 / 10) - np.exp(-jvect * x2 / 10)
    term2 = np.exp(-jvect / 10) - np.exp(-jvect)
    diff = term1 - x3 * term2
    
    gradient = np.zeros(x.shape[0], dtype='d')
    gradient[0] = np.sum(2 * diff * (-jvect / 10 * np.exp(-jvect * x1 / 10)))
    gradient[1] = np.sum(2 * diff * (jvect / 10 * np.exp(-jvect * x2 / 10)))
    gradient[2] = np.sum(2 * diff * (-term2))
    return gradient.flatten()



def problem246(x, lambda_noise):
    """
    Compute the objective function for Problem 246 with optional noise.

    Args:
        x (np.ndarray): Input array of size 3 reshaped to a column vector [x1, x2, x3].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3 = x[0], x[1], x[2]
    function_value = 100 * (x3 - ((x1 + x2) / 2)**2)**2 + (1 - x1)**2 + (1 - x2)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem246(x):
    """
    Compute the gradient of the objective function for Problem 246.

    Args:
        x (np.ndarray): Input array of size 3 reshaped to a column vector [x1, x2, x3].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3].
    """
    x1, x2, x3 = x[0], x[1], x[2]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = 200 * (x3 - ((x1 + x2) / 2)**2) * (-x1 - x2) * 0.5 - 2 * (1 - x1)
    gradient[1] = 200 * (x3 - ((x1 + x2) / 2)**2) * (-x1 - x2) * 0.5 - 2 * (1 - x2)
    gradient[2] = 200 * (x3 - ((x1 + x2) / 2)**2)
    return gradient

def problem255(x, lambda_noise):
    """
    Compute the objective function for Problem 255 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    function_value = (
        100 * (x2 - x1**2) + (1 - x1)**2
        + 90 * (x4 - x3**2) + (1 - x3)**2
        + 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
        + 19.8 * (x2 - 1) * (x4 - 1)
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem255(x):
    """
    Compute the gradient of the objective function for Problem 255.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3, df/dx4].
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = -200 * x1 - 2 * (1 - x1)
    gradient[1] = 100 + 20.2 * (x2 - 1) + 19.8 * (x4 - 1)
    gradient[2] = -180 * x3 - 2 * (1 - x3)
    gradient[3] = 90 + 20.2 * (x4 - 1) + 19.8 * (x2 - 1)
    return gradient

def problem256(x, lambda_noise):
    """
    Compute the objective function for Problem 256 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    function_value = (
        (x1 + 10 * x2)**2
        + 5 * (x3 - x4)**2
        + (x2 - 2 * x3)**4
        + 10 * (x1 - x4)**4
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem256(x):
    """
    Compute the gradient of the objective function for Problem 256.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3, df/dx4].
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = 2 * (x1 + 10 * x2) + 40 * (x1 - x4)**3
    gradient[1] = 20 * (x1 + 10 * x2) + 4 * (x2 - 2 * x3)**3
    gradient[2] = 10 * (x3 - x4) - 8 * (x2 - 2 * x3)**3
    gradient[3] = -10 * (x3 - x4) - 40 * (x1 - x4)**3
    return gradient

def problem258(x, lambda_noise):
    """
    Compute the objective function for Problem 258 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    function_value = (
        100 * (x2 - x1**2)**2
        + (1 - x1)**2
        + 90 * (x4 - x3**2)**2
        + (1 - x3)**2
        + 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
        + 19.8 * (x2 - 1) * (x4 - 1)
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise


def derivative_problem258(x):
    """
    Compute the gradient of the objective function for Problem 258.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector [x1, x2, x3, x4].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3, df/dx4].
    """
    x = x.reshape(-1, 1)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    grad = np.zeros(x.shape[0], dtype="d")
    grad[0] = 200 * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1)
    grad[1] = 200 * (x2 - x1**2) + 20.2 * (x2 - 1) + 19.8 * (x4 - 1)
    grad[2] = 180 * (x4 - x3**2) * (-2 * x3) - 2 * (1 - x3)
    grad[3] = 180 * (x4 - x3**2) + 20.2 * (x4 - 1) + 19.8 * (x2 - 1)
    return grad


def problem260(x, lambda_noise):
    """
    Compute the objective function for Problem 260 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 representing [x1, x2, x3, x4].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    function_value = (
        100 * (x2 - x1**2)**2
        + (1 - x1)**2
        + 90 * (x4 - x3**2)**2
        + (1 - x3)**2
        + 9.9 * ((x2 - 1) + (x4 - 1))**2
        + 0.2 * ((x2 - 1)**2 + (x4 - 1)**2)
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem260(x):
    """
    Compute the gradient of the objective function for Problem 260.

    Args:
        x (np.ndarray): Input array of size 4 representing [x1, x2, x3, x4].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3, df/dx4].
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = 200 * (x2 - x1**2) * (-2 * x1) - 2 * (1 - x1)
    gradient[1] = (
        200 * (x2 - x1**2)
        + 19.8 * ((x2 - 1) + (x4 - 1))
        + 0.4 * (x2 - 1)
    )
    gradient[2] = 180 * (x4 - x3**2) * (-2 * x3) - 2 * (1 - x3)
    gradient[3] = (
        180 * (x4 - x3**2)
        + 19.8 * ((x2 - 1) + (x4 - 1))
        + 0.4 * (x4 - 1)
    )
    return gradient

def problem261(x, lambda_noise):
    """
    Compute the objective function for Problem 261 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 representing [x1, x2, x3, x4].
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    function_value = (
        (np.exp(x1) - x2)**4
        + 100 * (x2 - x3)**6
        + (np.tan(x3 - x4))**4
        + x1**8
        + (x4 - 1)**2
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem261(x):
    """
    Compute the gradient of the objective function for Problem 261.

    Args:
        x (np.ndarray): Input array of size 4 representing [x1, x2, x3, x4].

    Returns:
        np.ndarray: Gradient vector [df/dx1, df/dx2, df/dx3, df/dx4].
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = 4 * np.exp(x1) * (np.exp(x1) - x2)**3 + 8 * x1**7
    gradient[1] = -4 * (np.exp(x1) - x2)**3 + 600 * (x2 - x3)**5
    gradient[2] = (
        -600 * (x2 - x3)**5
        + 4 * (np.tan(x3 - x4))**3 * (1 / np.cos(x3 - x4))**2
    )
    gradient[3] = (
        -4 * (np.tan(x3 - x4))**3 * (1 / np.cos(x3 - x4))**2
        + 2 * (x4 - 1)
    )
    return gradient


def problem266(x, lambda_noise):
    """
    Compute the objective function for Problem 266 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.0426149, 0.0352053, 0.0878058, 0.0330812, 0.0580924, 0.649704, 0.344144, -0.627443, 0.001828, -0.224783]).reshape(-1, 1)
    D = np.array([2.34659, 2.84048, 1.13888, 3.02286, 1.72139, 0.153917, 0.290577, -0.159378, 54.6910, -0.444873]).reshape(-1, 1)
    B = np.array([[0.354033, -0.0230349, -0.211938, -0.0554288, 0.220429],
                  [-0.0230349, 0.29135, -0.00180333, -0.111141, 0.0485461],
                  [-0.211938, -0.00180333, 0.815808, -0.133538, -0.38067],
                  [-0.0554288, -0.111141, -0.133538, 0.389198, -0.131586],
                  [0.220429, 0.0485461, -0.38067, -0.131586, 0.534706]])
    C = np.array([[-0.564255, 0.392417, -0.404979, 0.927589, -0.0735083],
                  [0.535493, 0.658799, -0.636666, -0.681091, -0.869487],
                  [0.586387, 0.289826, 0.854402, 0.789312, 0.949721],
                  [0.608734, 0.984915, 0.375699, 0.239547, 0.463136],
                  [0.774227, 0.325421, -0.151719, 0.448051, 0.149926],
                  [-0.435033, -0.688583, 0.222278, -0.524653, 0.413248],
                  [0.759468, -0.627795, 0.0403142, 0.724666, -0.0182537],
                  [-0.152448, -0.546437, 0.484134, 0.353951, 0.887866],
                  [-0.821772, -0.53412, -0.798498, -0.658572, 0.662362],
                  [0.819831, -0.910632, -0.480344, -0.871758, -0.978666]])
    F = A + np.dot(C, x) + 0.5 * ((x.T @ B) @ x) * D
    noise = np.random.normal(0, lambda_noise)
    function_value = F.T @ F
    return function_value[0][0] + noise

def derivative_problem266(x):
    """
    Compute the gradient of the objective function for Problem 266.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.array([0.0426149, 0.0352053, 0.0878058, 0.0330812, 0.0580924, 0.649704, 0.344144, -0.627443, 0.001828, -0.224783]).reshape(-1, 1)
    D = np.array([2.34659, 2.84048, 1.13888, 3.02286, 1.72139, 0.153917, 0.290577, -0.159378, 54.6910, -0.444873]).reshape(-1, 1)
    B = np.array([[0.354033, -0.0230349, -0.211938, -0.0554288, 0.220429],
                  [-0.0230349, 0.29135, -0.00180333, -0.111141, 0.0485461],
                  [-0.211938, -0.00180333, 0.815808, -0.133538, -0.38067],
                  [-0.0554288, -0.111141, -0.133538, 0.389198, -0.131586],
                  [0.220429, 0.0485461, -0.38067, -0.131586, 0.534706]])
    C = np.array([[-0.564255, 0.392417, -0.404979, 0.927589, -0.0735083],
                  [0.535493, 0.658799, -0.636666, -0.681091, -0.869487],
                  [0.586387, 0.289826, 0.854402, 0.789312, 0.949721],
                  [0.608734, 0.984915, 0.375699, 0.239547, 0.463136],
                  [0.774227, 0.325421, -0.151719, 0.448051, 0.149926],
                  [-0.435033, -0.688583, 0.222278, -0.524653, 0.413248],
                  [0.759468, -0.627795, 0.0403142, 0.724666, -0.0182537],
                  [-0.152448, -0.546437, 0.484134, 0.353951, 0.887866],
                  [-0.821772, -0.53412, -0.798498, -0.658572, 0.662362],
                  [0.819831, -0.910632, -0.480344, -0.871758, -0.978666]])
    F = A + np.dot(C, x) + 0.5 * ((x.T @ B) @ x) * D
    gradient = np.zeros((x.shape[0], 1)).reshape(1, -1)
    for i in range(10):
        gradient += 2 * F[i] * (C[i] + 0.5 * ((x.T @ (B + B.T) * D[i])))
    return gradient.flatten()

def problem267(x, lambda_noise):
    """
    Compute the objective function for Problem 267 with optional noise.

    Args:
        x (np.ndarray): Input array of size 5 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    zvect = 0.1 * np.arange(1, 12)
    yvect = np.exp(-zvect) - 5 * np.exp(-10 * zvect) + 3 * np.exp(4 * zvect)
    function_value = np.sum((x3 * np.exp(-x1 * zvect) - x4 * np.exp(-x2 * zvect) + 3 * np.exp(-x5 * zvect) - yvect)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem267(x):
    """
    Compute the gradient of the objective function for Problem 267.

    Args:
        x (np.ndarray): Input array of size 5 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    zvect = 0.1 * np.arange(1, 12)
    yvect = np.exp(-zvect) - 5 * np.exp(-10 * zvect) + 3 * np.exp(4 * zvect)
    gradient = np.zeros_like(x, dtype='d')
    for z, y in zip(zvect, yvect):
        gradient[0] += 2 * (x3 * np.exp(-x1 * z) - x4 * np.exp(-x2 * z) + 3 * np.exp(-x5 * z) - y) * (-x3 * z * np.exp(-x1 * z))
        gradient[1] += 2 * (x3 * np.exp(-x1 * z) - x4 * np.exp(-x2 * z) + 3 * np.exp(-x5 * z) - y) * (x4 * z * np.exp(-x2 * z))
        gradient[2] += 2 * (x3 * np.exp(-x1 * z) - x4 * np.exp(-x2 * z) + 3 * np.exp(-x5 * z) - y) * np.exp(-x1 * z)
        gradient[3] += 2 * (x3 * np.exp(-x1 * z) - x4 * np.exp(-x2 * z) + 3 * np.exp(-x5 * z) - y) * -np.exp(-x2 * z)
        gradient[4] += 2 * (x3 * np.exp(-x1 * z) - x4 * np.exp(-x2 * z) + 3 * np.exp(-x5 * z) - y) * (-3 * z * np.exp(-x5 * z))
    return gradient.flatten()

def problem271(x, lambda_noise):
    """
    Compute the objective function for Problem 271 with optional noise.

    Args:
        x (np.ndarray): Input array of size 6.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum([(16 - (i + 1)) * (x[i] - 1)**2 for i in range(6)])
    noise = np.random.normal(0, lambda_noise)
    return 10 * function_value + noise

def derivative_problem271(x):
    """
    Compute the gradient of the objective function for Problem 271.

    Args:
        x (np.ndarray): Input array of size 6.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.array([10 * (16 - (i + 1)) * 2 * (x[i] - 1) for i in range(6)])
    return gradient



def problem272(x, lambda_noise):
    """
    Compute the objective function for Problem 272 with optional noise.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3, x4, x5, x6 = x
    tvect = np.arange(1, 14) / 10
    yvect = np.exp(-tvect) - 5 * np.exp(-10 * tvect) + 3 * np.exp(-4 * tvect)
    function_value = np.sum((x4 * np.exp(-x1 * tvect) - x5 * np.exp(-x2 * tvect) + x6 * np.exp(-x3 * tvect) - yvect)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem272(x):
    """
    Compute the gradient of the objective function for Problem 272.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2, x3, x4, x5, x6 = x
    tvect = np.arange(1, 14) / 10
    yvect = np.exp(-tvect) - 5 * np.exp(-10 * tvect) + 3 * np.exp(-4 * tvect)
    gradient = np.zeros_like(x, dtype='d')
    for t, y in zip(tvect, yvect):
        diff = x4 * np.exp(-x1 * t) - x5 * np.exp(-x2 * t) + x6 * np.exp(-x3 * t) - y
        gradient[0] += 2 * diff * (-x4 * t * np.exp(-x1 * t))
        gradient[1] += 2 * diff * (x5 * t * np.exp(-x2 * t))
        gradient[2] += 2 * diff * (-x6 * t * np.exp(-x3 * t))
        gradient[3] += 2 * diff * np.exp(-x1 * t)
        gradient[4] += 2 * diff * -np.exp(-x2 * t)
        gradient[5] += 2 * diff * np.exp(-x3 * t)
    return gradient

def problem273(x, lambda_noise):
    """
    Compute the objective function for Problem 273 with optional noise.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum([(16 - (i + 1)) * (x[i] - 1)**2 for i in range(6)])
    noise = np.random.normal(0, lambda_noise)
    return 10 * function_value + 10 * function_value**2 + noise

def derivative_problem273(x):
    """
    Compute the gradient of the objective function for Problem 273.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    function_value = np.sum([(16 - (i + 1)) * (x[i] - 1)**2 for i in range(6)])
    for i in range(6):
        gradient[i] = 10 * (16 - (i + 1)) * 2 * (x[i] - 1)
        gradient[i] += 20 * (16 - (i + 1)) * 2 * (x[i] - 1) * function_value
    return gradient

def problem274(x, lambda_noise):
    """
    Compute the objective function for Problem 274 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A2 = np.array([[1, 1 / 2], [1 / 2, 1 / 3]])
    noise = np.random.normal(0, lambda_noise)
    return (x.T @ A2 @ x).item() + noise

def derivative_problem274(x):
    """
    Compute the gradient of the objective function for Problem 274.

    Args:
        x (np.ndarray): Input array of size 2 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    A2 = np.array([[1, 1 / 2], [1 / 2, 1 / 3]])
    return (x.T @ (A2 + A2.T)).flatten()

def problem275(x, lambda_noise):
    """
    Compute the objective function for Problem 275 with optional noise.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A4 = np.array([[1, 1 / 2, 1 / 3, 1 / 4],
                   [1 / 2, 1 / 3, 1 / 4, 1 / 5],
                   [1 / 3, 1 / 4, 1 / 5, 1 / 6],
                   [1 / 4, 1 / 5, 1 / 6, 1 / 7]])
    noise = np.random.normal(0, lambda_noise)
    return (x.T @ A4 @ x).item() + noise

def derivative_problem275(x):
    """
    Compute the gradient of the objective function for Problem 275.

    Args:
        x (np.ndarray): Input array of size 4 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    A4 = np.array([[1, 1 / 2, 1 / 3, 1 / 4],
                   [1 / 2, 1 / 3, 1 / 4, 1 / 5],
                   [1 / 3, 1 / 4, 1 / 5, 1 / 6],
                   [1 / 4, 1 / 5, 1 / 6, 1 / 7]])
    return (x.T @ (A4 + A4.T)).flatten()

def problem276(x, lambda_noise):
    """
    Compute the objective function for Problem 276 with optional noise.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A6 = hilbert(6)
    noise = np.random.normal(0, lambda_noise)
    return (x.T @ A6 @ x).item() + noise

def derivative_problem276(x):
    """
    Compute the gradient of the objective function for Problem 276.

    Args:
        x (np.ndarray): Input array of size 6 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    A6 = hilbert(6)
    return (x.T @ (A6 + A6.T)).flatten()


def problem281(x, lambda_noise):
    """
    Compute the objective function for Problem 281 with optional noise.

    Args:
        x (np.ndarray): Input array of size 10 representing the variables.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value raised to the power of 1/3 with noise.
    """
    j = np.arange(1, 11)
    function_value = np.sum((j**3) * (x - 1)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value**(1 / 3) + noise

def derivative_problem281(x):
    """
    Compute the gradient of the objective function for Problem 281.

    Args:
        x (np.ndarray): Input array of size 10 representing the variables.

    Returns:
        np.ndarray: Gradient vector.
    """
    j = np.arange(1, 11)
    function_value = np.sum((j**3) * (x - 1)**2)
    gradient = (j**3) * 2 * (x - 1)
    gradient *= (1 / 3) * function_value**(-2 / 3)
    return gradient

def problem282(x, lambda_noise):
    """
    Compute the objective function for Problem 282 with optional noise.

    Args:
        x (np.ndarray): Input array of size N representing the variables.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = 0.0
    # Match original computation of the boundary terms
    function_value += (x[0] - 1)**2 + (x[-1] - 1)**2
    # Sum over exactly 9 pairs (from i=0 to i=8)
    for i in range(9):
        j = i + 1
        function_value += 10 * (10 - j) * (x[i]**2 - x[i + 1])**2

    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem282(x):
    """
    Compute the gradient of the objective function for Problem 282.

    Args:
        x (np.ndarray): Input array of size N representing the variables.

    Returns:
        np.ndarray: Gradient vector.
    """
    n = x.shape[0]
    gradient = np.zeros(n, dtype='d')

    # Boundary derivatives
    gradient[0] += 2 * (x[0] - 1)
    gradient[-1] += 2 * (x[-1] - 1)

    # Sum over exactly 9 pairs (from i=0 to i=8)
    for i in range(9):
        j = i + 1
        # d/d(x[i]) of term => 20 * (10-j) * (x[i]^2 - x[i+1]) * (2*x[i])
        gradient[i]   += 20 * (10 - j) * (x[i]**2 - x[i + 1]) * (2*x[i])
        # d/d(x[i+1]) => -20*(10-j)*(x[i]^2 - x[i+1])
        gradient[i+1] += -20 * (10 - j) * (x[i]**2 - x[i + 1])

    return gradient




def problem283(x, lambda_noise):
    """
    Compute the objective function for Problem 283 with optional noise.

    Args:
        x (np.ndarray): Input array of size 10.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value cubed, with noise.
    """
    function_value = 0.0
    # Sum over exactly 10 terms (i=0..9), j=i+1
    for i in range(10):
        j = i + 1
        function_value += (j**3) * (x[i] - 1)**2

    noise = np.random.normal(0, lambda_noise)
    return function_value**3 + noise

def derivative_problem283(x):
    """
    Compute the gradient of the objective function for Problem 283.

    Args:
        x (np.ndarray): Input array of size 10.

    Returns:
        np.ndarray: Gradient vector.
    """
    n = x.shape[0]
    gradient = np.zeros(n, dtype='d')

    # First accumulate functionValue in the same loop
    function_value = 0.0
    for i in range(10):
        j = i + 1
        function_value += (j**3) * (x[i] - 1)**2
        # The partial derivative wrt x[i] of (x[i] - 1)^2 is 2(x[i] - 1)
        gradient[i] += (j**3) * 2 * (x[i] - 1)

    # Multiply entire gradient by 3*(function_value^2)
    gradient *= 3.0 * (function_value**2)

    return gradient

def problem286(x, lambda_noise):
    """
    Compute the objective function for Problem 286 with optional noise.

    Args:
        x (np.ndarray): Input array of size 20.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    xi, xi10 = x[:10], x[10:]
    function_value = np.sum(100 * (xi**2 - xi10)**2 + (xi - 1)**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem286(x):
    """
    Compute the gradient of the objective function for Problem 286.

    Args:
        x (np.ndarray): Input array of size 20.

    Returns:
        np.ndarray: Gradient vector.
    """
    xi, xi10 = x[:10], x[10:]
    gradient = np.zeros_like(x, dtype='d')
    gradient[:10] = 200 * (xi**2 - xi10) * 2 * xi + 2 * (xi - 1)
    gradient[10:] = -200 * (xi**2 - xi10)
    return gradient

def problem287(x, lambda_noise):
    """
    Compute the objective function for Problem 287 with optional noise.

    Args:
        x (np.ndarray): Input array of size 20.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = 0
    for i in range(5):
        function_value += (
            100 * (x[i]**2 + x[i + 5])**2
            + (x[i] - 1)**2
            + 90 * (x[i + 10]**2 + x[i + 15])**2
            + (x[i + 10] - 1)**2
            + 10.1 * ((x[i + 5] - 1)**2 + (x[i + 15] - 1)**2)
            + 19.8 * (x[i + 5] - 1) * (x[i + 15] - 1)
        )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem287(x):
    """
    Compute the gradient of the objective function for Problem 287.

    Args:
        x (np.ndarray): Input array of size 20.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    for i in range(5):
        gradient[i] += 200 * (x[i]**2 + x[i + 5]) * 2 * x[i] + 2 * (x[i] - 1)
        gradient[i + 5] += 200 * (x[i]**2 + x[i + 5]) + 20.2 * (x[i + 5] - 1) + 19.8 * (x[i + 15] - 1)
        gradient[i + 10] += 180 * (x[i + 10]**2 + x[i + 15]) * 2 * x[i + 10] + 2 * (x[i + 10] - 1)
        gradient[i + 15] += 180 * (x[i + 10]**2 + x[i + 15]) + 20.2 * (x[i + 15] - 1) + 19.8 * (x[i + 5] - 1)
    return gradient

def problem288(x, lambda_noise):
    """
    Compute the objective function for Problem 288 with optional noise.

    Args:
        x (np.ndarray): Input array of size 20.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = 0
    for i in range(5):
        function_value += (
            (x[i] + 10 * x[i + 5])**2
            + 5 * (x[i + 10] - x[i + 15])**2
            + (x[i + 5] - 2 * x[i + 10])**4
            + 10 * (x[i] - x[i + 15])**4
        )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem288(x):
    """
    Compute the gradient of the objective function for Problem 288.

    Args:
        x (np.ndarray): Input array of size 20.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    for i in range(5):
        gradient[i] += 2 * (x[i] + 10 * x[i + 5]) + 40 * (x[i] - x[i + 15])**3
        gradient[i + 5] += 20 * (x[i] + 10 * x[i + 5]) + 4 * (x[i + 5] - 2 * x[i + 10])**3
        gradient[i + 10] += 10 * (x[i + 10] - x[i + 15]) - 8 * (x[i + 5] - 2 * x[i + 10])**3
        gradient[i + 15] += -10 * (x[i + 10] - x[i + 15]) - 40 * (x[i] - x[i + 15])**3
    return gradient

def problem289(x, lambda_noise):
    """
    Compute the objective function for Problem 289 with optional noise.

    Args:
        x (np.ndarray): Input array of size 30.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = 1 - np.exp(-np.sum(x**2) / 60)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem289(x):
    """
    Compute the gradient of the objective function for Problem 289.

    Args:
        x (np.ndarray): Input array of size 30.

    Returns:
        np.ndarray: Gradient vector.
    """
    function_value = np.exp(-np.sum(x**2) / 60)
    return 2 * x / 60 * function_value




def problem290(x, lambda_noise):
    """
    Compute the objective function for Problem 290 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed squared matrix product with noise.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 3))
    noise = np.random.normal(0, lambda_noise)
    matrix_product = (x.T @ A @ x)**2
    return matrix_product[0][0] + noise

def derivative_problem290(x):
    """
    Compute the gradient of the objective function for Problem 290.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 3))
    return (2 * (x.T @ A @ x) * (x.T @ (A + A.T))).flatten()

def problem291(x, lambda_noise):
    """
    Compute the objective function for Problem 291 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed squared matrix product with noise.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 11))
    noise = np.random.normal(0, lambda_noise)
    matrix_product = (x.T @ A @ x)**2
    return matrix_product[0][0] + noise

def derivative_problem291(x):
    """
    Compute the gradient of the objective function for Problem 291.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 11))
    return (2 * (x.T @ A @ x) * (x.T @ (A + A.T))).flatten()

def problem292(x, lambda_noise):
    """
    Compute the objective function for Problem 292 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed squared matrix product with noise.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 31))
    noise = np.random.normal(0, lambda_noise)
    matrix_product = (x.T @ A @ x)**2
    return matrix_product[0][0] + noise

def derivative_problem292(x):
    """
    Compute the gradient of the objective function for Problem 292.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 31))
    return (2 * (x.T @ A @ x) * (x.T @ (A + A.T))).flatten()

def problem293(x, lambda_noise):
    """
    Compute the objective function for Problem 293 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed squared matrix product with noise.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 51))
    noise = np.random.normal(0, lambda_noise)
    matrix_product = (x.T @ A @ x)**2
    return matrix_product[0][0] + noise

def derivative_problem293(x):
    """
    Compute the gradient of the objective function for Problem 293.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = np.diag(np.arange(1, 51))
    return (2 * (x.T @ A @ x) * (x.T @ (A + A.T))).flatten()

def problem294(x, lambda_noise):
    """
    Compute the objective function for Problem 294 with optional noise.

    Args:
        x (np.ndarray): Input array of size 6 or greater.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum([100 * (x[k + 1] - x[k]**2)**2 + (1 - x[k])**2 for k in range(5)])
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem294(x):
    """
    Compute the gradient of the objective function for Problem 294.

    Args:
        x (np.ndarray): Input array of size 6 or greater.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    for k in range(5):
        gradient[k] += -200 * (x[k + 1] - x[k]**2) * 2 * x[k] - 2 * (1 - x[k])
        gradient[k + 1] += 200 * (x[k + 1] - x[k]**2)
    return gradient










def problem295(x, lambda_noise):
    """
    Compute the objective function for Problem 295 with optional noise.

    Args:
        x (np.ndarray): Input array of size 10.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem295(x):
    """
    Compute the gradient of the objective function for Problem 295.

    Args:
        x (np.ndarray): Input array of size 10.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    gradient[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    gradient[1:] += 200 * (x[1:] - x[:-1]**2)
    return gradient

def problem296(x, lambda_noise):
    """
    Compute the objective function for Problem 296 with optional noise.

    Args:
        x (np.ndarray): Input array of size 16.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem296(x):
    """
    Compute the gradient of the objective function for Problem 296.

    Args:
        x (np.ndarray): Input array of size 16.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    gradient[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    gradient[1:] += 200 * (x[1:] - x[:-1]**2)
    return gradient

def problem297(x, lambda_noise):
    """
    Compute the objective function for Problem 297 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem297(x):
    """
    Compute the gradient of the objective function for Problem 297.

    Args:
        x (np.ndarray): Input array of size 30.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    gradient[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    gradient[1:] += 200 * (x[1:] - x[:-1]**2)
    return gradient

def problem298(x, lambda_noise):
    """
    Compute the objective function for Problem 298 with optional noise.

    Args:
        x (np.ndarray): Input array of size 50 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem298(x):
    """
    Compute the gradient of the objective function for Problem 298.

    Args:
        x (np.ndarray): Input array of size 50.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    gradient[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    gradient[1:] += 200 * (x[1:] - x[:-1]**2)
    return gradient

def problem299(x, lambda_noise):
    """
    Compute the objective function for Problem 299 with optional noise.

    Args:
        x (np.ndarray): Input array of size 100 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem299(x):
    """
    Compute the gradient of the objective function for Problem 299.

    Args:
        x (np.ndarray): Input array of size 100.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')
    gradient[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    gradient[1:] += 200 * (x[1:] - x[:-1]**2)
    return gradient

def problem300(x, lambda_noise):
    """
    Compute the objective function for Problem 300 using a tridiagonal matrix with optional noise.

    Args:
        x (np.ndarray): Input array of size 20 reshaped to a column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(20, 20)).toarray()
    A[0, 0] = 1
    noise = np.random.normal(0, lambda_noise)
    matrix_product = x.T @ A @ x
    return matrix_product[0, 0] - 2 * x[0, 0] + noise

def derivative_problem300(x):
    """
    Compute the gradient of the objective function for Problem 300.

    Args:
        x (np.ndarray): Input array of size 20 reshaped to a column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(20, 20)).toarray()
    A[0, 0] = 1
    gradient = (x.T @ (A + A.T)).flatten()
    gradient[0] -= 2
    return gradient



def problem301(x, lambda_noise):
    """
    Compute the objective function for Problem 301 using a tridiagonal matrix with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 50.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(50, 50)).toarray()
    A[0, 0] = 1
    noise = np.random.normal(0, lambda_noise)
    matrix_product = x.T @ A @ x
    return matrix_product[0, 0] - 2 * x[0, 0] + noise

def derivative_problem301(x):
    """
    Compute the gradient of the objective function for Problem 301.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 50.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(50, 50)).toarray()
    A[0, 0] = 1
    gradient = (x.T @ (A + A.T)).flatten()
    gradient[0] -= 2
    return gradient

def problem302(x, lambda_noise):
    """
    Compute the objective function for Problem 302 using a tridiagonal matrix with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 100.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(100, 100)).toarray()
    A[0, 0] = 1
    noise = np.random.normal(0, lambda_noise)
    matrix_product = x.T @ A @ x
    return matrix_product[0, 0] - 2 * x[0, 0] + noise

def derivative_problem302(x):
    """
    Compute the gradient of the objective function for Problem 302.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 100.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(100, 100)).toarray()
    A[0, 0] = 1
    gradient = (x.T @ (A + A.T)).flatten()
    gradient[0] -= 2
    return gradient

def problem303(x, lambda_noise):
    """
    Compute the objective function for Problem 303.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 20.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    indices = np.arange(1, 21).reshape(-1, 1)
    function_value = (
        np.sum(x**2)
        + (np.sum(0.5 * indices * x))**2
        + (np.sum(0.5 * indices * x))**4
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem303(x):
    """
    Compute the gradient of the objective function for Problem 303.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 20.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    indices = np.arange(1, 21).reshape(-1, 1)
    gradient = 2 * x + 0.5 * indices * 2 * np.sum(0.5 * indices * x) + 0.5 * indices * 4 * (np.sum(0.5 * indices * x))**3
    return gradient.flatten()

def problem304(x, lambda_noise):
    """
    Compute the objective function for Problem 304.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 50.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    indices = np.arange(1, 51).reshape(-1, 1)
    function_value = (
        np.sum(x**2)
        + (np.sum(0.5 * indices * x))**2
        + (np.sum(0.5 * indices * x))**4
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem304(x):
    """
    Compute the gradient of the objective function for Problem 304.

    Args:
        x (np.ndarray): Input array reshaped to a column vector of size 50.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)
    indices = np.arange(1, 51).reshape(-1, 1)
    gradient = 2 * x + 0.5 * indices * 2 * np.sum(0.5 * indices * x) + 0.5 * indices * 4 * (np.sum(0.5 * indices * x))**3
    return gradient.flatten()


def problem305(x, lambda_noise):
    """
    Compute the objective function for Problem 305 with optional noise.

    Args:
        x (np.ndarray): Input array reshaped to column vector.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    indices = np.arange(1, 101).reshape(-1, 1)  # Create range 1 to 100 as a column vector
    term1 = np.sum(x**2)
    term2 = (np.sum(0.5 * indices * x))**2
    term3 = (np.sum(0.5 * indices * x))**4
    function_value = term1 + term2 + term3
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem305(x):
    """
    Compute the gradient of the objective function for Problem 305.

    Args:
        x (np.ndarray): Input array reshaped to column vector.

    Returns:
        np.ndarray: Gradient vector.
    """
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    indices = np.arange(1, 101).reshape(-1, 1)  # Create range 1 to 100 as a column vector
    common_term = np.sum(0.5 * indices * x)  # Common summation term
    grad_term1 = 2 * x
    grad_term2 = 0.5 * indices * 2 * common_term
    grad_term3 = 0.5 * indices * 4 * (common_term**3)
    gradient = grad_term1 + grad_term2 + grad_term3
    return gradient.flatten()



def problem307(x, lambda_noise):
    """
    Compute the objective function for Problem 307 with optional noise.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    function_value = 0
    for i in range(10):
        j = i + 1
        yi = 2 + 2 * j  # Compute the target value
        prediction = np.exp(j * x[0]) + np.exp(j * x[1])  # Predicted value
        function_value += (yi - prediction)**2  # Sum squared error
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem307(x):
    """
    Compute the gradient of the objective function for Problem 307.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    gradient = np.zeros_like(x, dtype='d')  # Initialize gradient vector
    for i in range(10):
        j = i + 1
        yi = 2 + 2 * j  # Target value
        prediction = np.exp(j * x[0]) + np.exp(j * x[1])  # Predicted value
        error = yi - prediction
        gradient[0] += 2 * error * (-j * np.exp(j * x[0]))  # Partial derivative w.r.t x[0]
        gradient[1] += 2 * error * (-j * np.exp(j * x[1]))  # Partial derivative w.r.t x[1]
    return gradient





def problem308(x, lambda_noise):
    """
    Compute the objective function for Problem 308.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (
        (x1**2 + x2**2 + x1 * x2)**2
        + np.sin(x1)**2
        + np.cos(x2)**2
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem308(x):
    """
    Compute the gradient of the objective function for Problem 308.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2 = x[0], x[1]
    gradient = np.zeros_like(x, dtype='d')
    gradient[0] = 2 * (x1**2 + x2**2 + x1 * x2) * (2 * x1 + x2) + 2 * np.sin(x1) * np.cos(x1)
    gradient[1] = 2 * (x1**2 + x2**2 + x1 * x2) * (2 * x2 + x1) - 2 * np.cos(x2) * np.sin(x2)
    return gradient


def problem309(x, lambda_noise):
    """
    Compute the objective function for Problem 309.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (
        1.41 * x1**4 - 12.76 * x1**3 + 39.91 * x1**2 - 51.93 * x1 + 24.37
        + (x2 - 3.9)**2
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem309(x):
    """
    Compute the gradient of the objective function for Problem 309.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2 = x[0], x[1]
    gradient = np.zeros_like(x)
    gradient[0] = 4 * 1.41 * x1**3 - 3 * 12.76 * x1**2 + 2 * 39.91 * x1 - 51.93
    gradient[1] = 2 * (x2 - 3.9)
    return gradient

def problem311(x, lambda_noise):
    """
    Compute the objective function for Problem 311.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem311(x):
    """
    Compute the gradient of the objective function for Problem 311.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2 = x[0], x[1]
    gradient = np.zeros_like(x)
    gradient[0] = 4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7)
    gradient[1] = 2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    return gradient

def problem312(x, lambda_noise):
    """
    Compute the objective function for Problem 312.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = (
        (x1**2 + 12 * x2 - 1)**2
        + (49 * x1**2 + 49 * x2**2 + 84 * x1 + 2324 * x2 - 681)**2
    )
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem312(x):
    """
    Compute the gradient of the objective function for Problem 312.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2 = x[0], x[1]
    gradient = np.zeros_like(x)
    term1 = x1**2 + 12 * x2 - 1
    term2 = 49 * x1**2 + 49 * x2**2 + 84 * x1 + 2324 * x2 - 681
    gradient[0] = 4 * x1 * term1 + 2 * term2 * (98 * x1 + 84)
    gradient[1] = 24 * term1 + 2 * term2 * (98 * x2 + 2324)
    return gradient

def problem314(x, lambda_noise):
    """
    Compute the objective function for Problem 314.

    Args:
        x (np.ndarray): Input array of size 2.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    g1x = -(x1**2) / 4 - x2**2 + 1
    h1x = x1 - 2 * x2 + 1
    function_value = (x1 - 2)**2 + (x2 - 1)**2 + 0.04 / g1x + (h1x**2) / 0.2
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem314(x):
    """
    Compute the gradient of the objective function for Problem 314.

    Args:
        x (np.ndarray): Input array of size 2.

    Returns:
        np.ndarray: Gradient vector.
    """
    x1, x2 = x[0], x[1]
    g1x = -(x1**2) / 4 - x2**2 + 1
    h1x = x1 - 2 * x2 + 1
    gradient = np.zeros_like(x)
    gradient[0] = 2 * (x1 - 2) - 0.04 / (g1x**2) * (-0.5 * x1) + 2 * h1x / 0.2
    gradient[1] = 2 * (x2 - 1) - 0.04 / (g1x**2) * (-2 * x2) + 2 * h1x * (-2 / 0.2)
    return gradient


def problem333(x, lambda_noise):
    """
    Compute the objective function for Problem 333.

    Args:
        x (np.ndarray): Input array of size 3.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    a = np.array([4, 5.75, 7.5, 24, 32, 48, 72, 96])
    y = np.array([72.1, 65.6, 55.9, 17.1, 9.8, 4.5, 1.3, 0.6])
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # Same exact expression: sum of ((y - x1*exp(-x2*a) - x3)/y)^2
    function_value = np.sum(((y - x1*np.exp(-x2*a) - x3) / y)**2)

    # Add noise
    noise = np.array(np.random.normal(0, lambda_noise), dtype='d')
    return function_value + noise

def derivative_problem333(x):
    """
    Compute the gradient of the objective function for Problem 333.

    Args:
        x (np.ndarray): Input array of size 3.

    Returns:
        np.ndarray: Gradient vector of size 3.
    """
    n = x.shape[0]
    # As in original code, reshape x and the gradient
    x = x.reshape(-1, 1)
    grad = np.zeros(n, dtype='d').reshape(-1, 1)

    a = np.array([4, 5.75, 7.5, 24, 32, 48, 72, 96])
    y = np.array([72.1, 65.6, 55.9, 17.1, 9.8, 4.5, 1.3, 0.6])
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # Replicate the same partial derivative sums
    tmp = 2 * ((y - x1*np.exp(-x2*a) - x3) / y)
    grad[0] += np.sum(tmp * (-np.exp(-x2*a) / y))
    grad[1] += np.sum(tmp * (-x1*np.exp(-x2*a)*(-a / y)))
    grad[2] += np.sum(tmp * (-1 / y))

    return grad.reshape(-1,)  # Flatten to shape (N,)

def problem334(x, lambda_noise):
    """
    Compute the objective function for Problem 334.

    Args:
        x (np.ndarray): Input array of size 3.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3 = x.flatten()
    ui = np.arange(1, 16).reshape(-1, 1)
    vi = 16 - ui
    wi = np.minimum(ui, vi)
    y = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]).reshape(-1, 1)
    residuals = y - (x1 + ui / (x2 * vi + x3 * wi))
    function_value = np.sum(residuals**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem334(x):
    """
    Compute the gradient of the objective function for Problem 334.

    Args:
        x (np.ndarray): Input array of size 3.

    Returns:
        np.ndarray: Gradient vector of size 3.
    """
    x1, x2, x3 = x.flatten()
    ui = np.arange(1, 16).reshape(-1, 1)
    vi = 16 - ui
    wi = np.minimum(ui, vi)
    y = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]).reshape(-1, 1)
    residuals = y - (x1 + ui / (x2 * vi + x3 * wi))

    grad = np.zeros_like(x)
    grad[0] = np.sum(-2 * residuals)
    grad[1] = np.sum(-2 * residuals * (-vi * ui / (x2 * vi + x3 * wi)**2))
    grad[2] = np.sum(-2 * residuals * (-wi * ui / (x2 * vi + x3 * wi)**2))
    return grad


def problem351(x, lambda_noise):
    """
    Compute the objective function for Problem 351.

    Args:
        x (np.ndarray): Input array of size 4.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2, x3, x4 = x.flatten()
    b = np.array([7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32]).reshape(-1, 1)
    a = np.array([0.0, 0.000428, 0.00100, 0.00161, 0.00209, 0.00348, 0.00525]).reshape(-1, 1)
    
    numerator = x1**2 + a * x2**2 + (a**2) * x3**2
    denominator = 1 + a * x4**2
    residuals = (numerator / denominator - b) / b
    function_value = 10**4 * np.sum(residuals**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem351(x):
    """
    Compute the gradient of the objective function for Problem 351.

    Args:
        x (np.ndarray): Input array of size 4.

    Returns:
        np.ndarray: Gradient vector of size 4.
    """
    x1, x2, x3, x4 = x.flatten()
    b = np.array([7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32]).reshape(-1, 1)
    a = np.array([0.0, 0.000428, 0.00100, 0.00161, 0.00209, 0.00348, 0.00525]).reshape(-1, 1)
    
    numerator = x1**2 + a * x2**2 + (a**2) * x3**2
    denominator = 1 + a * x4**2
    residuals = (numerator / denominator - b) / b
    
    grad = np.zeros_like(x).reshape(-1, 1)
    grad[0] = 10**4 * 2 * np.sum(residuals * (2 * x1 / denominator) / b)
    grad[1] = 10**4 * 2 * np.sum(residuals * (2 * x2 * a / denominator) / b)
    grad[2] = 10**4 * 2 * np.sum(residuals * (2 * x3 * a**2 / denominator) / b)
    grad[3] = 10**4 * 2 * np.sum(residuals * (-numerator / denominator**2) * (2 * x4 * a) / b)
    return grad.flatten()

def problem352(x, lambda_noise):
    """
    Compute the objective function for Problem 352.

    Args:
        x (np.ndarray): Input array of size 4.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    function_value = 0.0
    # Original loop from i=0 to 19
    for i in range(20):
        ti = 0.2 * (i + 1)
        function_value += (x1 + x2*ti - np.exp(ti))**2 + (x3 + x4*np.sin(ti) - np.cos(ti))**2

    noise = np.array(np.random.normal(0, lambda_noise), dtype='d')
    return function_value + noise

def derivative_problem352(x):
    """
    Compute the gradient of the objective function for Problem 352.

    Args:
        x (np.ndarray): Input array of size 4.

    Returns:
        np.ndarray: Gradient vector of size 4.
    """
    n = x.shape[0]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    grad = np.zeros(n, dtype='d').reshape(-1, 1)

    # Original loop from i=0 to 19
    for i in range(20):
        ti = 0.2 * (i + 1)
        # Partial derivatives wrt x1, x2, x3, x4
        tmp1 = (x1 + x2*ti - np.exp(ti))
        tmp2 = (x3 + x4*np.sin(ti) - np.cos(ti))
        grad[0] += 2 * tmp1
        grad[1] += 2 * tmp1 * ti
        grad[2] += 2 * tmp2
        grad[3] += 2 * tmp2 * np.sin(ti)

    return grad.reshape(-1,)

def problem370(x, lambda_noise):
    """
    Compute the objective function for Problem 370.

    Args:
        x (np.ndarray): Input array of size 6.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1 = x[0]
    x2 = x[1]

    function_value = x1**2 + (x2 - x1**2 - 1)**2
    # Loop i=2..30
    for i in range(2, 31):
        # inner sum for k=1..5
        inside = 0.0
        for k in range(1, 6):
            j = k + 1
            inside += (j - 1) * x[k] * ((i - 1)/29)**(j - 2)

        # minusSquare sum for k=0..5
        minus_square = 0.0
        for k in range(0, 6):
            j = k + 1
            minus_square += x[k] * ((i - 1)/29)**(j - 1)

        inside -= minus_square**2 - 1
        function_value += inside**2

    noise = np.array(np.random.normal(0, lambda_noise), dtype='d')
    return function_value + noise

def derivative_problem370(x):
    """
    Compute the gradient of the objective function for Problem 370.

    Args:
        x (np.ndarray): Input array of size 6.

    Returns:
        np.ndarray: Gradient vector of size 6.
    """
    n = x.shape[0]
    grad = np.zeros(n, dtype='d').reshape(-1, 1)

    x1 = x[0]
    x2 = x[1]

    # Derivatives of the initial terms x1^2 + (x2 - x1^2 -1)^2
    grad[0] = 2*x1 + 2*(x2 - x1**2 - 1)*(-2*x1)
    grad[1] = 2*(x2 - x1**2 - 1)

    # Outer loop i=2..30
    for i in range(2, 31):
        # Compute the 'inside' expression
        inside = 0.0
        for k in range(1, 6):
            j = k + 1
            inside += (j - 1) * x[k] * ((i - 1)/29)**(j - 2)

        minus_square = 0.0
        for k in range(0, 6):
            j = k + 1
            minus_square += x[k] * ((i - 1)/29)**(j - 1)

        inside -= minus_square**2 - 1

        # Gradient of inside^2 => 2*inside * derivative(inside)
        # For each kder in 0..5
        for kder in range(6):
            j = kder + 1
            if kder >= 1:
                # derivative wrt x[kder]: (j-1)*((i-1)/29)^(j-2) - 2*minusSquare*((i-1)/29)^(j-1)
                grad[kder] += 2*inside * (
                    (j - 1)*((i - 1)/29)**(j - 2)
                    - 2*minus_square*((i - 1)/29)**(j - 1)
                )
            else:
                # kder == 0 => only the -2*minusSquare(...) part
                grad[kder] += 2*inside * (
                    -2*minus_square*((i - 1)/29)**(j - 1)
                )

    return grad.reshape(-1,)


def problem371(x, lambda_noise):
    """
    Compute the objective function for Problem 371.

    Args:
        x (np.ndarray): Input array of size 9.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x1, x2 = x[0], x[1]
    function_value = x1**2 + (x2 - x1**2 - 1)**2

    for i in range(2, 31):
        t = (i - 1) / 29
        inner_sum = sum((j + 1) * x[j] * t**(j - 1) for j in range(1, 9))
        square_term = sum(x[j] * t**j for j in range(9))**2
        residual = inner_sum - square_term - 1
        function_value += residual**2

    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem371(x):
    """
    Compute the gradient of the objective function for Problem 371.

    Args:
        x (np.ndarray): Input array of size 9.

    Returns:
        np.ndarray: Gradient vector of size 9.
    """
    x1, x2 = x[0], x[1]
    grad = np.zeros_like(x)
    grad[0] = 2 * x1 - 4 * x1 * (x2 - x1**2 - 1)
    grad[1] = 2 * (x2 - x1**2 - 1)

    for i in range(2, 31):
        t = (i - 1) / 29
        inner_sum = sum((j + 1) * x[j] * t**(j - 1) for j in range(1, 9))
        square_term = sum(x[j] * t**j for j in range(9))
        residual = inner_sum - square_term**2 - 1

        for j in range(9):
            grad[j] += 2 * residual * ((j + 1) * t**(j - 1) if j >= 1 else 0 - 2 * square_term * t**j)
    return grad

def problem379(x, lambda_noise):
    """
    Compute the objective function for Problem 379.

    Args:
        x (np.ndarray): Input array of size 11.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(-1, 1)
    y = np.array([1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725,
                  0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724,
                  0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495,
                  0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
                  0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632,
                  0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581,
                  0.428, 0.292, 0.162, 0.098, 0.054]).reshape(-1, 1)
    t = 0.1 * (np.arange(1, 66).reshape(-1, 1) - 1)

    terms = [
        x[0] * np.exp(-x[4] * t),
        x[1] * np.exp(-x[5] * (t - x[8])**2),
        x[2] * np.exp(-x[6] * (t - x[9])**2),
        x[3] * np.exp(-x[7] * (t - x[10])**2)
    ]

    prediction = sum(terms)
    residuals = y - prediction
    function_value = np.sum(residuals**2)
    noise = np.random.normal(0, lambda_noise)
    return function_value + noise

def derivative_problem379(x):
    """
    Compute the gradient of the objective function for Problem 379.

    Args:
        x (np.ndarray): Input array of size 11.

    Returns:
        np.ndarray: Gradient vector of size 11.
    """
    x = x.reshape(-1, 1)
    y = np.array([1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725,
                  0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724,
                  0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495,
                  0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
                  0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632,
                  0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581,
                  0.428, 0.292, 0.162, 0.098, 0.054]).reshape(-1, 1)
    t = 0.1 * (np.arange(1, 66).reshape(-1, 1) - 1)

    prediction = (
        x[0] * np.exp(-x[4] * t) +
        x[1] * np.exp(-x[5] * (t - x[8])**2) +
        x[2] * np.exp(-x[6] * (t - x[9])**2) +
        x[3] * np.exp(-x[7] * (t - x[10])**2)
    )
    residuals = y - prediction

    grad = np.zeros_like(x)
    grad[0] = -2 * np.sum(residuals * np.exp(-x[4] * t))
    grad[4] = 2 * np.sum(residuals * x[0] * t * np.exp(-x[4] * t))
    grad[5] = 2 * np.sum(residuals * x[1] * (t - x[8])**2 * np.exp(-x[5] * (t - x[8])**2))
    return grad.flatten()



def problem391(x, lambda_noise):
    """
    Compute the objective function for Problem 391.

    Args:
        x (np.ndarray): Input array of size 30.
        lambda_noise (float): Standard deviation of Gaussian noise to add to the result.

    Returns:
        float: The computed objective function value with noise.
    """
    x = x.reshape(1, -1)
    i = np.arange(1, 31).reshape(-1, 1)  # Row indices
    j = np.arange(1, 31).reshape(1, -1)  # Column indices

    # Compute the V matrix
    V = np.sqrt(x**2 + i / j)

    # Calculate the function value components
    sin_log_V = np.sin(np.log(V))
    cos_log_V = np.cos(np.log(V))
    xi = x.reshape(-1, 1)
    funV = 420 * xi + (i - 15)**3 + np.sum(V * (sin_log_V**5 + cos_log_V**5), axis=1).reshape(-1, 1)

    # Add Gaussian noise
    noise = np.random.normal(0, lambda_noise)
    return np.sum(funV) + noise

def derivative_problem391(x):
    """
    Compute the gradient of the objective function for Problem 391.

    Args:
        x (np.ndarray): Input array of size 30.

    Returns:
        np.ndarray: Gradient vector of size 30.
    """
    x = x.reshape(-1)
    n = x.size
    i = np.arange(1, 31).reshape(-1, 1)  # Row indices
    j = np.arange(1, 31).reshape(1, -1)  # Column indices

    # Initialize the gradient vector
    grad = np.zeros_like(x, dtype=float)

    # Compute the V matrix
    V = np.sqrt(x**2 + i / j)

    # Calculate the derivatives
    sin_log_V = np.sin(np.log(V))
    cos_log_V = np.cos(np.log(V))
    dV_dx = x / np.sqrt(x**2 + i / j)

    for k in range(n):
        grad[k] += 420  # Constant term
        # Derivative contribution from V terms
        grad[k] += np.sum(
            dV_dx[:, k] * (sin_log_V[:, k]**5 + cos_log_V[:, k]**5)
            + V[:, k] * (
                5 * sin_log_V[:, k]**4 * cos_log_V[:, k] * (-1 / V[:, k]) * dV_dx[:, k]
                - 5 * cos_log_V[:, k]**4 * sin_log_V[:, k] * (-1 / V[:, k]) * dV_dx[:, k]
            )
        )

    return grad












