import numpy as np
from mpmath import mp
import time


def MinimaxComp(a, b, alpha, epsilon, depth, margin):
    # Step 1: Obtain the optimal set of degrees Mdegs from ComputeMinMultDegs
    Mdegs = [15, 15, 15, 15, 27]
    basis = [lambda x: x**i for i in range(6)]

    # Step 2: Initialize the polynomial and error for the first degree
    p = MP(1 - epsilon, 1, Mdegs[0])
    tau = ME(1 - epsilon, 1, Mdegs[0]) + margin

    # Step 3: Perform the homomorphic comparison for each degree
    for i in range(1,len(Mdegs)):
        degree = Mdegs[i]
        p_i = MP(tau-epsilon,tau ,degree)
        tau = ME(tau-epsilon,tau, degree) + margin
        p = compose_polynomials(p,p_i) 

    result = (evaluate_polynomial(p, (a-b))+1)/2
    return result

# Additional functions for polynomial approximation and error calculation
def MP(D, d, degree):
    
    polynomial = np.zeros(27)
    for i in range(degree):
        polynomial[i]=1

    return polynomial

def compose_polynomials(poly1, poly2):
    result_deg = len(poly1) * (len(poly2) - 1)  
    result_poly = [0] * (result_deg + 1)  

    for i, coeff1 in enumerate(poly1):
        for j, coeff2 in enumerate(poly2):
            result_poly[i + j] += coeff1 * coeff2

    return result_poly

def evaluate_polynomial(coefficients, x):
    mp.dps = 3
    result = mp.fsum(mp.mpf(coeff) * (mp.mpf(x) ** (i+1)) for i, coeff in enumerate(coefficients))
    return round(float(result),3) 

def ME(D, d, degree):

    return 0.1


if __name__ == "__main__":
    L_a = [0.11,0.111,-0.111, 3, 0,1,0,-10,9,0.1, 0.1,-0.5,0.5,-39, -200032,23193,2.5, -1, 0.75, -5, 8, 15, -0.3, 0, -2, 4, 0.8, 7, -12, -3, 0.2, -0.9, 6, -0.6, 1, 0]
    L_b = [0.1,0.112,-0.112, 3, 1,0,0, 10,2.5, 1, 0.75, 5, -8, -15, -0.3, 0, -2, -4, -0.8, -7, 12, 3, -0.2, 0.9, -6, 0.6, -1, 0,6,0.2,-0.1, 0.5,0.4, 200,-13025, 238408]

    alpha = 20
    epsilon = 21
    depth = 21
    margin = 0.0001

    total_time = 0

    for a, b in zip(L_a, L_b):
        start_time = time.time()
        result = MinimaxComp(a, b, alpha, epsilon, depth, margin)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if(result==0.5):
            print(f"result = {result} ## {a} = {b}")
        elif(result<0.5):
            print(f"result = {result} ## {a} < {b}")
        elif(result>0.5):
            print(f"result = {result} ## {a} > {b}")
            
        print(f"Running Time for the operation: {elapsed_time:.6f} seconds - {time.time()}")

        #print(f"MinimaxComp({a}, {b}): {result}")

    print(f"Total Running Time: {total_time:.6f} seconds")