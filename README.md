# Constrained Optimization with Newton's Method

## Overview
This project implements a constrained optimization problem using Newton's method with a logarithmic barrier function. The goal is to minimize the given function while ensuring that constraints are satisfied.

## Problem Definition
The function to be minimized is:

\[ f(x_1, x_2, x_3, x_4, x_5, t) = (2x_1 + 2x_2 - x_3 + 4x_4 + 2x_5)

subject to the constraints:
- \( x_1 + x_3 + \sqrt{x_2^2 + x_4^2} < 0 \)
- \( -8 - x_1 + 3x_2 + 2x_3 - x_5 > 0 \)

## Implementation
The solution is implemented in Python using:
- **Gradient Computation**: Analytical derivatives are computed for the function.
- **Hessian Matrix**: Numerical approximation of the second derivatives.
- **Newton's Method**: Iteratively updates variables using second-order optimization.
- **Backtracking Line Search**: Ensures step size is chosen optimally.
- **Log Barrier Method**: Helps handle constraints smoothly.

## Key Functions
### `f(x1, x2, x3, x4, x5, t)`
Computes the function value with barrier terms.

### `generate_random_values()`
Generates valid initial values that satisfy constraints.

### `gradient_f(variables, t)`
Computes the gradient of the function.

### `hessian_f(variables, t, h=1e-5)`
Approximates the Hessian matrix using finite differences.

### `backtracking(variables, steplength, gradient, t, alpha=0.1, betta=0.5)`
Performs backtracking line search to find the optimal step size.

### `newton_method(init_var, t, error=1e-6, max_steps=10000)`
Applies Newton's method to iteratively minimize the function.

