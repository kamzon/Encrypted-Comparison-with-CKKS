import math
import numpy as np
from Pyfhel import Pyfhel
import time

def MinimaxComp(ctxt_a, ctxt_b, alpha, epsilon, depth, margin, HE):
    Mdegs = [15, 15, 15, 15, 27]
    basis = [lambda x: x**i for i in range(6)]

    p = MP(1 - epsilon, 1, Mdegs[0])
    tau = ME(1 - epsilon, 1, Mdegs[0], HE) + margin

    for i in range(1, len(Mdegs)):
        degree = Mdegs[i]
        p_i = MP(tau - epsilon, tau, degree)
        tau = ME(tau - epsilon, tau, degree, HE) + margin
        p = compose_polynomials(p, p_i, HE)

    c_x = ctxt_a - ctxt_b
    result = evaluate_polynomial(p, c_x, HE)
    return result

def MP(D, d, degree):
    polynomial = np.zeros(27)
    for i in range(degree):
        polynomial[i] = 1

    return polynomial

def compose_polynomials(poly1, poly2, HE):
    result_deg = len(poly1) + len(poly2) - 1
    result_poly = [0] * result_deg

    for i, coeff1 in enumerate(poly1):
        for j, coeff2 in enumerate(poly2):
            result_poly[i + j] += coeff1 * coeff2

    return result_poly

def evaluate_polynomial(coefficients, ctxt_x, HE):
    c_result = []

    for i, coeff in enumerate(coefficients):
        if i == 0:
            monomial = HE.encrypt(1)
        else:
            monomial = ctxt_x
            for j in range(1, i):
                monomial = monomial * ctxt_x

        monomial = monomial * HE.encode(coeff)
        c_result.append(monomial)

    return c_result

def ME(D, d, degree, HE):
    return HE.encrypt([0.1])

if __name__ == "__main__":
    L_a = [1]
    L_b = [0]

    alpha = 20
    epsilon = 21
    depth = 21
    margin = 0.0001

    HE = Pyfhel()
    ckks_params = {
        'scheme': 'CKKS',
        'n': 2 ** 15,
        'scale': 2 ** 30,
        'qi_sizes': [60] + 10 * [30] + [60]
    }

    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.rotateKeyGen()

    total_time = 0

    for a, b in zip(L_a, L_b):
        start_time = time.time()
        encrypted_a = HE.encrypt(a)
        encrypted_b = HE.encrypt(b)

        c_result = MinimaxComp(encrypted_a, encrypted_b, alpha, epsilon, depth, margin, HE)

        decrypted_result = HE.decrypt(c_result)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if decrypted_result == 0.5:
            print(f"{a} = {b}")
        elif decrypted_result < 0.5:
            print(f"{a} < {b}")
        elif decrypted_result > 0.5:
            print(f"{a} > {b}")

    print(f"Total Running Time: {total_time:.6f} seconds")
