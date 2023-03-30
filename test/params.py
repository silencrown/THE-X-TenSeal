import tenseal as ts
import time


start_time = time.time()

def renew_time(end_op=None):
    global start_time
    elapsed_time = time.time() - start_time
    print(f"{end_op} elapsed time: {elapsed_time:2f}s")
    start_time = time.time()

def bfv_test():
    context = ts.context(ts.SCHEME_TYPE.BFV, 
                        poly_modulus_degree=4096, 
                        plain_modulus=1032193)

    public_context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
    # public_context.is_private()
    # public_context.is_public()

    sk = public_context.secret_key()

    # the context will drop the secret-key at this point
    public_context.make_context_public()

    plain_vector = [60, 66, 73, 81, 90]
    encrypted_vector = ts.bfv_vector(context, plain_vector)
    renew_time()

    print("plaintext vector of size:", encrypted_vector.size())
    print("encrypted vector:", plain_vector)

    add_result = encrypted_vector + [1, 2, 3, 4, 5]
    print(add_result.decrypt())
    renew_time()

    sub_result = encrypted_vector - [1, 2, 3, 4, 5]
    print(sub_result.decrypt())
    renew_time()

    mul_result = encrypted_vector * [1, 2, 3, 4, 5]
    print(mul_result.decrypt())
    renew_time()

    encrypted_vector_tmp = ts.bfv_vector(context, [1,2,3,4,5])
    enc_mul_result = encrypted_vector_tmp * encrypted_vector
    print(enc_mul_result.decrypt())
    renew_time()

def ckks_test():
    """Encryption Parameters"""
    start_time = time.time()

    # controls precision of the fractional part
    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        # coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    renew_time(end_op="Initial")

    # encrypt the vector by using ckks
    plain_vector = [600.77, 6600.23, 7300.23, 8100.12, 900.56]
    encrypted_vector = ts.ckks_vector(context, plain_vector)
    renew_time(end_op="Encryption")

    # c + p
    tmp_vector = [1.1, 1.1, 1.1, 1.1, 1.1]
    add = encrypted_vector + tmp_vector
    renew_time(end_op="c + p")

    # c - p
    sub = encrypted_vector - tmp_vector
    renew_time(end_op="c - p")

    # c * p
    mul = encrypted_vector * tmp_vector
    renew_time(end_op="c * p")

    # c * c
    enc_mul = encrypted_vector * ts.ckks_vector(context, plain_vector)
    renew_time(end_op="c * c")

    # mul depth
    for i in range(16):
        encrypted_vector = encrypted_vector * ts.ckks_vector(context, [0.1, 0.1, 0.1, 0.1, 0.1])
        renew_time(end_op=f"mul {i}")
        print(encrypted_vector.decrypt())
        print("="*50)


if __name__ == "__main__":
    ckks_test()
