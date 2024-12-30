# -*- coding: utf-8 -*-
"""
Gradient Approximation Analysis - No Noise

This script computes and compares various gradient approximation methods on 
multiple optimization problems from the Schittkowski test set. The results 
are organized into Excel files for further analysis. 

Key features:
- Supports multiple sigma values for smoothing.
- Configurable number of function evaluations per coordinate.
- Results are categorized into gradient norm "buckets" to track optimization progress.
- This version of the script performs calculations without adding noise to the function evaluations.

Author: Marco Boresta
Refactored: 27/12/2024

Usage:
- Ensure the required Schittkowski functions and bucket definitions are available.
- Run the script to generate Excel files containing the results for the specified parameters.

Note:
This script specifically deals with the no-noise version of the gradient approximation analysis.
"""

import os
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from utils import evaluate_function_across_interval, compute_error_metrics
from gradient_approximation_methods import (
    GSG,
    cGSG,
    CFD,
    ACFD,
    NMXFD,
    FFD,
)
import schittkowski_functions as foo

# Parameters
SIGMA_LIST = [1e-2, 1e-5, 1e-8]  # List of sigma values to test
FUNC_EVAL_LIST = [3, 5, 9]       # Number of function evaluations per coordinate
BUCKET_LIST = list(range(8))     # Buckets to process

# Paths
BUCKETS_FILE = "buckets.pkl"
OUTPUT_FOLDER = "Results/no_noise/"

# Constants
SEED_START = 23  # Initial seed for reproducibility


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


def compute_gradient_approximations(problem_number, sigma, func_evals, bucket_dic, seed):
    """
    Compute gradient approximations for a specific problem.

    Args:
        problem_number (int): Problem index.
        sigma (float): Sigma value for gradient approximation.
        func_evals (int): Number of function evaluations per coordinate.
        bucket_dic (dict): Bucket dictionary with problem details.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Results of gradient approximations and their comparisons.
    """
    np.random.seed(seed)
    objective_function = getattr(foo, f"problem{problem_number}")
    derivative_function = getattr(foo, f"derivative_problem{problem_number}")
    t = bucket_dic[problem_number].reshape(1, -1)
    f_ob_orig = objective_function(t.reshape(-1,), 0)

    lambda_noise = abs(f_ob_orig * 0.00)  # Placeholder noise multiplier
    range_integral = [-2, 2]
    num_directions = int((func_evals - 1) * t.shape[1] / 2)

    # Define arguments for different methods
    args = {
        "NMXFD": (objective_function, func_evals, sigma, lambda_noise, range_integral, seed),
        "CFD": (objective_function, sigma, lambda_noise, seed),
        "GS": (objective_function, num_directions * 2, sigma, lambda_noise, seed),
        "GSC": (objective_function, num_directions, sigma, lambda_noise, seed),
    }

    # Compute gradients
    gradients = {
        "NMXFD": np.asarray(evaluate_function_across_interval(t, NMXFD, *args["NMXFD"])),
        "Real": np.asarray(evaluate_function_across_interval(t, derivative_function)),
        "FFD": np.asarray(evaluate_function_across_interval(t, FFD, *args["CFD"])),
        "CFD": np.asarray(evaluate_function_across_interval(t, CFD, *args["CFD"])),
        "GSG": np.asarray(evaluate_function_across_interval(t, GSG, *args["GS"])),
        "cGSG": np.asarray(evaluate_function_across_interval(t, cGSG, *args["GSC"])),
        "ACFD": np.asarray(evaluate_function_across_interval(t, ACFD, *args["NMXFD"])),
    }

    # Compute distances between approximations and the real derivative
    distances = {method: compute_error_metrics(gradients[method], gradients["Real"]) for method in gradients if method != "Real"}

    # Extract results
    return {method: distances[method][0] for method in distances}


def generate_results(sigma, func_evals, bucket_index, bucket_dic):
    """
    Process all problems in a specific bucket.

    Args:
        sigma (float): Sigma value for gradient approximation.
        func_evals (int): Number of function evaluations per coordinate.
        bucket_index (int): Index of the current bucket.
        bucket_dic (dict): Bucket dictionary with problem details.

    Returns:
        pd.DataFrame: DataFrame containing results for all problems in the bucket.
    """
    results = []
    seed = SEED_START

    for prob_number in tqdm(bucket_dic.keys(), desc=f"Processing Bucket {bucket_index}"):
        # print(prob_number)
        seed += 1
        prob_results = compute_gradient_approximations(prob_number, sigma, func_evals, bucket_dic, seed)
        prob_results.update({
            "Problem Number": prob_number,
            "Dimension": bucket_dic[prob_number].shape[0],
            "Function Evaluations": func_evals,
            "Sigma": sigma,
        })
        results.append(prob_results)

    return pd.DataFrame(results)


def save_results(df_results, filename):
    """
    Save results DataFrame to an Excel file.

    Args:
        df_results (pd.DataFrame): DataFrame containing results.
        filename (str): Name of the output file (without extension).
    """
    output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.xlsx")
    df_results.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    """
    Main function to execute the script.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load bucket definitions
    buckets = load_buckets(BUCKETS_FILE)

    # Process each combination of sigma, function evaluations, and bucket
    for sigma in SIGMA_LIST:
        for func_evals in FUNC_EVAL_LIST:
            for bucket_index in BUCKET_LIST:
                bucket_dic = buckets[bucket_index]
                filename = f"bucket{bucket_index}_N{func_evals}_sigma-{sigma}"
                print(f"Processing {filename}...")

                # Generate and save results
                df_results = generate_results(sigma, func_evals, bucket_index, bucket_dic)
                save_results(df_results, filename)


if __name__ == "__main__":
    main()
