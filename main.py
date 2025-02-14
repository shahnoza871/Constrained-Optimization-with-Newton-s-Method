import numpy as np
import random

def f(x1, x2, x3, x4, x5, t):

    return (
        (2*x1 + 2*x2 - x3 + 4*x4 + 2*x5)
        - (1/t)*(
            np.log(-x1 - x3 - np.sqrt(x2**2 + x4**2)) 
            + np.log(-8 - x1 + 3*x2 + 2*x3 - x5)
        )
    )

import numpy as np

def generate_random_values():
    i=0
    while True:
        # Generate random values for each variable
        x1 = np.random.uniform(-100, 100)
        x2 = np.random.uniform(-100, 100)
        x3 = np.random.uniform(-100, 100)
        x4 = np.random.uniform(-100, 100)
        x5 = np.random.uniform(-100, 100)
        
        if (x1 + x3 + np.sqrt(x2**2 + x4**2) < 0) and (-8 - x1 + 3*x2 + 2*x3 - x5>0):
            return x1, x2, x3, x4, x5
        elif i==1000000:
            print('Valid Initial Value NotFound')
            return 0
        i+=1


def gradient_f(variables, t):
    x1, x2, x3, x4, x5 = variables
    # Compute each partial derivative analytically
    term1 = -x1 - x3 - np.sqrt(x2**2 + x4**2)
    term2 = -8 - x1 + 3*x2 + 2*x3 - x5
    grad_x1 = 2 + (1/t) * (1 / term1 + 1 / term2)
    grad_x2 = 2 + (1/t) * (x2 / np.sqrt(x2**2 + x4**2) / term1 + 3 / term2)
    grad_x3 = -1 + (1/t) * (1 / term1 + 2 / term2)
    grad_x4 = 4 + (1/t) * (x4 / np.sqrt(x2**2 + x4**2) / term1)
    grad_x5 = 2 - (1/t) * (1 / term2)
    
    return np.array([grad_x1, grad_x2, grad_x3, grad_x4, grad_x5])

def gradlength(variables, t):
    grad = gradient_f(variables, t)
    return np.sum(grad**2)

def backtracking(variables, steplength, gradient, t, alpha=0.1, betta=0.5):
    # Backtracking line search
    while f(*(variables - steplength * gradient), t) > f(*variables, t) - alpha * steplength * gradlength(variables, t):
        steplength *= betta
    return steplength

def hessian_f(variables, t, h=1e-5):
    n = len(variables)
    hessian_matrix = np.zeros((n, n))
    
    # Compute second partial derivatives
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements: second derivative with respect to the same variable
                step = np.zeros_like(variables)
                step[i] = h
                hessian_matrix[i, j] = (
                    f(*(variables + step), t)
                    - 2 * f(*variables, t)
                    + f(*(variables - step), t)
                ) / (h**2)
            else:
                # Off-diagonal elements: mixed partial derivatives
                step_i = np.zeros_like(variables)
                step_j = np.zeros_like(variables)
                step_i[i] = h
                step_j[j] = h
                hessian_matrix[i, j] = (
                    f(*(variables + step_i + step_j), t)
                    - f(*(variables + step_i), t)
                    - f(*(variables + step_j), t)
                    + f(*variables, t)
                ) / (h**2)
    
    return hessian_matrix

def newton_method(init_var, t, error=1e-6, max_steps=10000):
    variables = np.array(init_var)
    gradient = gradient_f(variables, t)
    step = 0
    
    while True:
        hessian = hessian_f(variables, t)
        # Ensure the Hessian is not singular by adding a small identity matrix
        hessian_inv = np.linalg.inv(hessian + np.eye(len(variables)) * 1e-5)
        update = hessian_inv.dot(gradient)
        
        # Backtracking line search
        steplength = 1.0
        steplength = backtracking(variables, steplength, gradient, t)
        
        tmpr = variables-steplength * update
        x1, x2, x3, x4, x5 = tmpr
        if (x1 + x3 + np.sqrt(x2**2 + x4**2) < 0) and (x1 - 3*x2 - 2*x3 + x5 < -8):
            variables = tmpr
        elif step >= max_steps:
            break
        
        gradient = gradient_f(variables, t)
        step += 1
    
    return variables


t = 1.0 
minimals = []
interval = [-100, 100]
for j in range(10):
    initial = generate_random_values()
    x1, x2, x3, x4, x5 = newton_method(initial, t, error=10**(-3))
    func = f(x1, x2, x3, x4, x5, t)
    minimals.append(float(func))
    print(f"Minimal Function Value is {func} at ({x1, x2, x3, x4, x5}) ")
    # t*=1.2

print(f"Sorted values of minimals1: {sorted(minimals)[:10]}")

