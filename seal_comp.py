import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, Evaluator, CKKSEncoder, Plaintext, Ciphertext
import numpy as np
import time
import math
#from mpmath import mp

def setup_seal_context(poly_modulus_degree=32768):
    params = EncryptionParameters(seal.scheme_type.ckks)
    params.set_poly_modulus_degree(poly_modulus_degree)
    params.set_coeff_modulus(seal.CoeffModulus.Create(poly_modulus_degree, [60, 60, 60, 60,40,30,30,60,60,60,60, 60, 60,60,60, 60]))
    return SEALContext(params)


def initialize_keys(context):
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()  # Adjusted to directly access the secret key
    return keygen, public_key, secret_key

def MP_SEAL(coefficients, degree, encoder,encryptor, scale):
    # Initialize a polynomial with 27 zeros
    polynomial = np.zeros(27)

    # Set the coefficients up to the specified degree to 1
    for i in range(degree):
        polynomial[i] = 1

    return polynomial


def create_encryption_tools(context, public_key, secret_key):
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    encoder = CKKSEncoder(context)
    return encryptor, decryptor, evaluator, encoder  # Add decryptor here


def ME_SEAL(encoder, encryptor, scale):

    # Constant value to be returned
    constant_value = 0.1

    # Encode and encrypt the constant value
    encoded_value = encoder.encode(constant_value, scale)
    encrypted_value = encryptor.encrypt(encoded_value)

    return encrypted_value

def compose_polynomials_SEAL_simple(poly1, poly2, evaluator):
    # Determine the size of the result polynomial
    result_size = len(poly1) * (len(poly2) - 1) 

    # Initialize the result polynomial with zeros
    result_poly = [0] * result_size
    for i, coeff1 in enumerate(poly1):
        for j, coeff2 in enumerate(poly2):
            result_poly[i + j] += coeff1 * coeff2

    return result_poly

def evaluate_polynomial_SEAL(coefficients, ctxt_x, evaluator, encoder, decryptor, relin_keys, scale):
    # Initialize an encrypted result to zero
    plaintext_zero = encoder.encode(0.0, scale)
    ctxt_result = encryptor.encrypt(plaintext_zero)
 

    # Initialize a list to store intermediate results
    intermediate_results = []

    # Evaluate the polynomial at the encrypted point
    for i, coeff in enumerate(coefficients):
        # Ensure coeff is a plaintext value (float or int) for encoding
        if(coeff == 0):
            continue
        if isinstance(coeff, (float, int)):
            # Encode the coefficient as Plaintext 
            plaintext_coeff = encoder.encode(coeff, scale)
            #print(coeff)
            #return ctxt_x
        elif isinstance(coeff, seal.Plaintext):
            # If coeff is already a Plaintext, use it directly
            plaintext_coeff = coeff
        else:
            raise TypeError("Coefficient is not a valid type for encoding")
        if i== 26:
            break
        if i == 0:
            # Add the constant term

            intermediate_results.append(encryptor.encrypt(plaintext_coeff))
        else:
            # Calculate coeff * x^i and add it to the intermediate results
            temp_ctxt = ctxt_x

            for _ in range(1,i):  # Start from 0 up to i (inclusive)

                temp_ctxt = evaluator.multiply(temp_ctxt, ctxt_x)
                evaluator.relinearize_inplace(temp_ctxt, relin_keys)


            temp_ctxt = evaluator.multiply_plain(temp_ctxt, plaintext_coeff)
 
            intermediate_results.append(temp_ctxt)
            temp_ctxt = encryptor.encrypt(plaintext_zero)
            evaluator.rescale_to_next_inplace(temp_ctxt)

            # Clean temp_ctxt
            evaluator.rescale_to_next_inplace(temp_ctxt)
            temp_ctxt = evaluator.relinearize(temp_ctxt, relin_keys)
            evaluator.mod_switch_to_next_inplace(temp_ctxt)

    ctxt_result = intermediate_results[0]
    res=0

    for i in range(0,len(intermediate_results)):
        ctxt_result_copy = intermediate_results[i]
        decrypted_result = Plaintext()
        decryptor.decrypt(ctxt_result_copy, decrypted_result)
        result = encoder.decode(decrypted_result)
        res+= result[0]
        #print(f"result {i} is : ",round(result[0],6))
        
        #print(f"Scale of ctxt_result: {ctxt_result.scale()}")
        #print(f"Scale of intermediate_result[{i}]: {intermediate_results[i].scale()}")
        #ctxt_result.scale(2**int(math.log2(intermediate_results[i].scale())))
        
        #evaluator.add_inplace(ctxt_result, intermediate_results[i])

        #intermediate_results[i].scale(2**int(math.log2(intermediate_results[i].scale())))
        # if i <len(intermediate_results)-1:
        #     ctxt_result.scale(2**int(math.log2(intermediate_results[i+1].scale())))
        

        

        # 
        # ctxt_result = evaluator.relinearize(intermediate_results[i], relin_keys)
        # ctxt_result = evaluator.add(ctxt_result, intermediate_results[i])
        # evaluator.mod_switch_to_next_inplace(ctxt_result)
        
    print("sum is : ",sum)

    return res


def homomorphic_MinimaxComp(a, b, alpha, epsilon, depth, margin, context, encryptor, decryptor, evaluator, encoder, relin_keys):
    # Step 1: Obtain the optimal set of degrees Mdegs
    Mdegs = [15,15,15,15,27]

    # Encrypt inputs a and b
    # Define a scale for encoding
    scale = pow(2.0, 30)

    # Encode the float values
    plaintext_a = encoder.encode([a], scale)
    plaintext_b = encoder.encode([b], scale)
    #encod_diff = encoder.encode([a-b], scale)

    # Encrypt the encoded values
    encrypted_a = encryptor.encrypt(plaintext_a)
    encrypted_b = encryptor.encrypt(plaintext_b)

    encrypted_diff = Ciphertext()
    #encrypted_diff = encryptor.encrypt(encod_diff)
    # Compute a - b homomorphically and store the result in encrypted_diff
    encrypted_diff = evaluator.sub(encrypted_a,encrypted_b)

    # Step 2: Initialize the polynomial (p) and error (tau) for the first degree

    encrypted_p = MP_SEAL(1 - epsilon, Mdegs[0], encoder, encryptor, scale)

    encrypted_tau = ME_SEAL(encoder, encryptor, scale)  # Homomorphically compute tau

    # Step 3: Perform the homomorphic comparison for each degree
    for i in range(1, len(Mdegs)):
        degree = Mdegs[i]

        # Compute p_i homomorphically
        encrypted_p_i = MP_SEAL(1 - epsilon, Mdegs[0], encoder, encryptor, scale)

        # Update tau homomorphically
        encrypted_tau = ME_SEAL(encoder, encryptor, scale)

        # Update the polynomial p homomorphically by composing it with p_i
        encrypted_p = compose_polynomials_SEAL_simple(encrypted_p, encrypted_p_i, evaluator)

    # Evaluate the polynomial p at the encrypted difference (a - b) homomorphically
    result = evaluate_polynomial_SEAL(encrypted_p, encrypted_diff, evaluator, encoder, decryptor, relin_keys, scale)

    # Decrypt and decode the result
    # decrypted_result = Plaintext()
    # decryptor.decrypt(encrypted_result, decrypted_result)
    # result = encoder.decode(decrypted_result)

    #decryptor.decrypt(encrypted_diff, decrypted_result)
    #result = encoder.decode(decrypted_result)

    return (result+1)/2


# Example usage
context = setup_seal_context()
keygen, public_key, secret_key = initialize_keys(context)
encryptor, decryptor, evaluator, encoder = create_encryption_tools(context, public_key, secret_key)

# Generate relinearization keys
relin_keys = keygen.create_relin_keys()

# Define parameters for the comparison
alpha = 20
epsilon = 0.1  # Example epsilon value, adjust as needed
depth = 5  # Example depth, adjust as needed
margin = 0.0001  # Example margin, adjust as needed

# Example inputs for comparison
input_pairs = [
    (1.0001, 1),
    (1,1),
    (0,0),
    (10,1),
    (-10,0)    
    # Add more pairs as needed
]
total_time = 0

# Perform homomorphic comparisons
for a, b in input_pairs:
    start_time = time.time()
    result = homomorphic_MinimaxComp(a, b, alpha, epsilon, depth, margin, context, encryptor, decryptor, evaluator, encoder, relin_keys)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    
    if(round(result,4)==1):
        print(f"result = {round(result,4)} ## {a} = {b}")
    elif(round(result,4)<1):
        print(f"result = {round(result,4)} ## {a} < {b}")
    elif(round(result,4)>1):
        print(f"result = {round(result,4)} ## {a} > {b}")
        
    print(f"Running Time for the operation: {elapsed_time:.6f} ")

print(f"Total Running Time: {total_time:.6f} seconds")
                                                                                                                                                        