# Gradient Approximation with NMXFD

This repository contains Python scripts and resources for implementing and analyzing the **Normalized Mixed Finite Differences (NMXFD)** method for gradient approximation, as described in the paper:  
*A Mixed Finite Differences Scheme for Gradient Approximation*, **Journal of Optimization Theory and Applications**, 2022. DOI: [10.1007/s10957-021-01994-w](https://doi.org/10.1007/s10957-021-01994-w).

## Overview

NMXFD is a novel method for approximating the gradient of functions, particularly useful for derivative-free optimization (DFO) in both noise-free and noisy settings. This repository provides tools to evaluate and compare gradient approximation methods on benchmark problems from the Schittkowski test set.

### Features
- Gradient approximation for **noise-free** and **noisy** function evaluations.
- Support for multiple smoothing parameter ($\sigma$) values.
- Aggregation and statistical analysis of gradient estimation errors across multiple runs.
- Implementation of test functions from the **Schittkowski test set**.

---

## Benchmarks

NMXFD is compared against the following established gradient approximation methods:

1. **Finite Difference Methods**  
   - Forward Finite Differences (FFD) and Central Finite Differences (CFD).  
   These methods are standard approaches to gradient approximation in optimization. Details can be found in:  
     - *Polyak, B.T.: Introduction to Optimization, vol. 1. Inc., Publications Division, New York (1987)*

2. **Gaussian Smoothing Methods**  
   - Gaussian Smoothed Gradient (GSG) and Central Gaussian Smoothed Gradient (cGSG).  
   These are statistical approaches that approximate gradients by smoothing with a Gaussian kernel. References include:  
     - *Flaxman, A. D., Wang, A. X., & Smola, A. J. (2005). Fast Gradient-Free Methods for Convex Optimization*.  
     - *Nesterov, Y., & Spokoiny, V. (2017). Random Gradient-Free Minimization of Convex Functions*. Foundations of Computational Mathematics.

These methods serve as benchmarks for evaluating the performance of NMXFD in both noise-free and noisy settings.

---

## Repository Structure

### Scripts

1. **`gradient_approximation_no_noise.py`**  
   Computes and compares various gradient approximation methods for **noise-free settings**. Results are exported as Excel files.

2. **`gradient_approximation_noise.py`**  
   Similar to the above, but introduces noise into function evaluations.

3. **`aggregate_results_no_noise.py`**  
   Aggregates gradient approximation results from the noise-free experiments. Computes median log errors and generates summary tables.  
   **Uses**: `template_summary_file_no_noise.xlsx`.

4. **`aggregate_results_noise.py`**  
   Processes noisy results, computes median log errors and generates summary tables.   
   **Uses**: `template_summary_file_noise.xlsx`.

5. **`schittkowski_functions.py`**  
   Implements optimization test problems and their derivatives, including:
   - Quadratic functions
   - Polynomial functions
   - Functions with exponential or trigonometric terms  

6. **`gradient_approximation_methods.py`**  
   Implements NMXFD and other gradient estimation methods, including:
   - Forward Finite Differences (FFD)
   - Central Finite Differences (CFD)
   - Gaussian Smoothed Gradient (GSG)
   - Central Gaussian Smoothed Gradient (cGSG)

7. **`utils.py`**  
   Utility functions for:
   - Error computation (absolute, relative, angular)
   - Function evaluation across intervals.

---

## Excel Templates

- **`template_summary_file_no_noise.xlsx`**  
   Template used by `aggregate_results_no_noise.py` to organize and summarize results from noise-free experiments.

- **`template_summary_file_noise.xlsx`**  
   Template used by `aggregate_results_noise.py` for summarizing noisy experiments.

These templates are pre-formatted to accept median log error values and organize them by method, smoothing parameter ($\sigma$), and problem dimension.

---

## Notes on Code History and Replicability

The code in this repository is a **refactored version** of the original implementation written in 2019/2020. Significant efforts have been made to ensure **replicability** and **consistency** with the results presented in the referenced paper. The refactoring process was assisted by LLMs, and every change has been double-checked to ensure alignment between the published results and those obtained with the new code. However, you might encounter unusual comments, formatting, or variable names due to this process. While care has been taken to align the current implementation with the published methodology, there might be minor discrepancies.

If you encounter any such misalignment or issues, please feel free to report them!


---

## Usage Instructions

### Prerequisites
- Python 3.7+
- Required libraries: `numpy`, `scipy`, `pandas`, `openpyxl`

### Running Gradient Approximation
1. Configure parameters for smoothing ($\sigma$), bucket ranges, and noise levels.
2. Use:
   - `gradient_approximation_no_noise.py` for noise-free experiments.
   - `gradient_approximation_noise.py` for noisy settings.
3. Results will be saved as Excel files in the designated output folder.

### Aggregating Results
- Use `aggregate_results_no_noise.py` or `aggregate_results_noise.py` to summarize results from individual Excel files into median log error tables.
- Ensure the corresponding template (`template_summary_file_no_noise.xlsx` or `template_summary_file_noise.xlsx`) is in the same directory or accessible.

### Benchmark Test Problems
- `schittkowski_functions.py` provides implementations for the Schittkowski test set. These functions are used in gradient approximation scripts.

---

## Output Format

1. **Intermediate Results**  
   Excel files containing gradient norm "buckets," error metrics, and performance across multiple seeds.

2. **Summary Tables**  
   Final aggregated tables showing:
   - Median log errors for each method.
   - Performance categorized by smoothing parameters and problem dimensions.

---

## Paper Reference

If you use this repository in your work, please cite:

```bibtex
@article{Boresta2022,
  title={A Mixed Finite Differences Scheme for Gradient Approximation},
  author={Marco Boresta, Tommaso Colombo, Alberto De Santis, Stefano Lucidi},
  journal={Journal of Optimization Theory and Applications},
  volume={194},
  number={1},
  pages={1--24},
  year={2022},
  doi={10.1007/s10957-021-01994-w}
}
```

---

## Contact
For questions or further information, please contact:
marco.boresta@gmail.com