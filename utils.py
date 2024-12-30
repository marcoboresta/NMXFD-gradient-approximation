# -*- coding: utf-8 -*-
"""
Utility Functions for Gradient Approximation and Evaluation

This script contains utility functions to assist with gradient approximation and evaluation tasks. 
It includes methods for computing the distance between predicted and real gradient vectors and for evaluating 
a function across a range of input values.

Functions:
- compute_error_metrics: Computes absolute, relative, and angular errors between predicted and real gradient vectors.
- evaluate_function_across_interval: Evaluates a given function across a specified interval of input values.

Author: Marco
Created: Wed Jul 10, 2019
Refactored: [27/12/2024]
"""

import numpy as np

def compute_error_metrics(predicted_points, real_points):
    """
    Compute error metrics (percentage, angle) between predicted and real points.

    Args:
        predicted_points (np.ndarray): Predicted gradient vectors of shape (n, d).
        real_points (np.ndarray): True gradient vectors of shape (n, d).

    Returns:
        tuple: Percentage error (mean of relative errors as percentage) and angular error 
               (mean angle in degrees between predicted and real vectors), rounded to 14 decimal places.
    """
    # Compute the difference between predicted and real points
    diff = real_points - predicted_points

    # Initialize error lists
    absolute_errors = []
    percentage_errors = []
    angular_errors = []

    for i in range(real_points.shape[0]):
        # Compute the angle between predicted and real vectors
        numerator = np.dot(predicted_points[i], real_points[i])
        denominator = np.linalg.norm(predicted_points[i]) * np.linalg.norm(real_points[i])
        if denominator != 0:
            angle = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))  # Clip to handle numerical instability
        else:
            angle = 0  # Default to zero if denominator is zero
        angular_errors.append(np.degrees(angle))  # Convert radians to degrees

        # Compute absolute and percentage errors
        absolute_errors.append(np.linalg.norm(diff[i]))
        percentage_errors.append(np.linalg.norm(diff[i]) / np.linalg.norm(real_points[i]))

    # Compute mean values of errors
    percentage_mean = np.mean(percentage_errors) * 100  # Convert to percentage
    angular_mean = np.mean(angular_errors)

    # Return rounded error metrics
    return np.round(percentage_mean, 14), np.round(angular_mean, 14)


def evaluate_function_across_interval(interval, function, *args):
    """
    Evaluate a function across a specified interval of input values.

    Args:
        interval (iterable): Iterable of input values to evaluate the function.
        function (callable): The function to evaluate.
        *args: Additional arguments to pass to the function.

    Returns:
        list: List of function values corresponding to each input value in the interval.
    """
    results = []
    for value in interval:
        # Evaluate the function at the current value with additional arguments
        results.append(function(value, *args))
    return results
