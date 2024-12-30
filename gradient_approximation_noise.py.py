# -*- coding: utf-8 -*-
"""
Gradient Approximation Analysis - Noisy Setting

This script computes and compares various gradient approximation methods on 
multiple optimization problems from the Schittkowski test set under a noisy setting. 
Function evaluations are perturbed with a configurable percentage of noise, and 
the results are organized into Excel files for further analysis.

Key features:
- Supports multiple sigma values for smoothing.
- Configurable number of function evaluations per coordinate.
- Results are categorized into gradient norm "buckets" to track optimization progress.
- Handles additive noise in function evaluations to simulate real-world conditions.
- Aggregates results across 100 seeds to ensure robustness and reproducibility.

Author: Marco Boresta
Refactored: 27/12/2024

Usage:
- Ensure the required Schittkowski functions, bucket definitions, and utility functions are available.
- Configure noise level, sigma values, and function evaluations as needed.
- Run the script to generate Excel files containing the results for the specified parameters.

Note:
This script specifically deals with the noisy version of the gradient approximation analysis, introducing
a percentage of noise into function evaluations to simulate practical scenarios.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from utils import evaluate_function_across_interval, compute_error_metrics
from gradient_approximation_methods import (
    NMXFD,
    FFD,
    CFD,
    GSG,
    cGSG,
    ACFD,
)
import schittkowski_functions as foo

PERC_NOISE = 0.001     
SIGMA_LIST = [10**-1, 10**-2, 10**-3]
FUNC_EVAL_LIST = [9,5,13]
BUCKET_LIST = [0,7,4,1,2,3,5,6]

# Paths
BUCKETS_FILE = "buckets.pkl"
OUTPUT_FOLDER = "Results/noise/"


def load_buckets(buckets_file):
    """
    Load bucket definitions from a pickle file.

    Args:
        buckets_file (str): Path to the pickle file containing bucket definitions.

    Returns:
        list: A list of dictionaries for each bucket.
    """
    with open(buckets_file, "rb") as f:
        return pickle.load(f)


def process_problem(problem_number, bucket_dic, sigma, func_evals, perc_noise):
    """
    Process a single optimization problem and compute mean relative errors.

    Args:
        problem_number (int): Index of the problem to process.
        bucket_dic (dict): Dictionary containing the bucket data.
        sigma (float): Sigma value for gradient approximation.
        func_evals (int): Number of function evaluations per coordinate.
        perc_noise (float): Percentage noise for function evaluations.

    Returns:
        dict: Dictionary containing the mean relative errors for all methods.
    """
    objective_function = getattr(foo, f"problem{problem_number}")
    derivative_function = getattr(foo, f"derivative_problem{problem_number}")
    t = bucket_dic[problem_number].reshape(1, -1)

    # Calculate noise level
    lambda_noise = 10**-3

    # Define arguments for different methods
    args_nmxfd = (objective_function, func_evals, sigma, lambda_noise, [-2, 2])
    args_fcd = (objective_function, sigma, lambda_noise)
    args_gs = (objective_function, (func_evals - 1) * t.shape[1], sigma, lambda_noise)
    args_gsc = (objective_function, (func_evals - 1) * t.shape[1] // 2, sigma, lambda_noise)

    # Aggregate errors across seeds
    errors = {
        "NMXFD": [],
        "FFD": [],
        "CFD": [],
        "GSG": [],
        "cGSG": [],
        "ACFD": [],
    }

    for seed in range(100):
        np.random.seed(seed)

        # Compute gradients for all methods
        nmxfd_points = evaluate_function_across_interval(t, NMXFD, *args_nmxfd, seed)[0].reshape(1,-1)
        fcd_points = evaluate_function_across_interval(t, CFD, *args_fcd, seed)[0].reshape(1,-1)
        fd_points = evaluate_function_across_interval(t, FFD, *args_fcd, seed)[0].reshape(1,-1)
        gs_points = evaluate_function_across_interval(t, GSG, *args_gs, seed)[0].reshape(1,-1)
        gsc_points = evaluate_function_across_interval(t, cGSG, *args_gsc, seed)[0].reshape(1,-1)
        acfd_points = evaluate_function_across_interval(t, ACFD, *args_nmxfd, seed)[0].reshape(1,-1)
        real_derivative_points = evaluate_function_across_interval(t, derivative_function)[0].reshape(1,-1)

        # Compute relative errors
        errors["NMXFD"].append(compute_error_metrics(nmxfd_points, real_derivative_points)[0])
        errors["FFD"].append(compute_error_metrics(fd_points, real_derivative_points)[0])
        errors["CFD"].append(compute_error_metrics(fcd_points, real_derivative_points)[0])
        errors["GSG"].append(compute_error_metrics(gs_points, real_derivative_points)[0])
        errors["cGSG"].append(compute_error_metrics(gsc_points, real_derivative_points)[0])
        errors["ACFD"].append(compute_error_metrics(acfd_points, real_derivative_points)[0])

    # Compute mean relative error for each method
    mean_errors = {method: np.mean(errors[method]) for method in errors}
    mean_errors.update({
        "Problem number": problem_number,
        "Dimension of problem": t.shape[1],
        "N": func_evals,
        "Lambda": lambda_noise,
    })

    return mean_errors


def process_bucket(bucket_number, bucket_dic, sigma, func_evals, perc_noise, output_folder):
    """
    Process all problems in a bucket and save results to an Excel file.

    Args:
        bucket_number (int): Bucket index.
        bucket_dic (dict): Dictionary containing the bucket data.
        sigma (float): Sigma value for gradient approximation.
        func_evals (int): Number of function evaluations per coordinate.
        perc_noise (float): Percentage noise for function evaluations.
        output_folder (str): Path to save the output Excel file.

    Returns:
        None
    """
    print(f"Processing bucket {bucket_number} with sigma={sigma} and N={func_evals}...")
    results = []

    for prob_number in tqdm(bucket_dic.keys(), desc=f"Bucket {bucket_number}"):
        mean_errors = process_problem(prob_number, bucket_dic, sigma, func_evals, perc_noise)
        results.append(mean_errors)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save results to Excel
    filename = f"bucket{bucket_number}_N{func_evals}_sigma-{sigma}_noise-{perc_noise}.xlsx"
    output_path = os.path.join(output_folder, filename)
    df_results.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    """
    Main function to process all buckets, sigmas, and function evaluations.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    buckets = load_buckets(BUCKETS_FILE)

    for sigma in SIGMA_LIST:
        for func_evals in FUNC_EVAL_LIST:
            for bucket_number in BUCKET_LIST:
                bucket_dic = buckets[bucket_number]
                process_bucket(bucket_number, bucket_dic, sigma, func_evals, PERC_NOISE, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
