from phe import paillier
import time 

def paillier_test():
    # Initial
    public_key,private_key = paillier.generate_paillier_keypair()
    message_list = [3.1415926,100,-4.6e-12]

    # Encryption
    time_start_enc = time.time()
    encrypted_message_list = [public_key.encrypt(m) for m in message_list]
    time_end_enc = time.time()
    print(f"Enc result: {encrypted_message_list[0].ciphertext}")
    print(f"Enc time: {time_end_enc-time_start_enc:.2f} s")

    # Decryption
    time_start_dec = time.time()
    decrypted_message_list = [private_key.decrypt(c) for c in encrypted_message_list]
    time_end_dec = time.time()
    print(f"Dec time: {time_end_dec-time_start_dec:.2f} s")
    print(f"Dec data: {decrypted_message_list}")

    a,b,c = encrypted_message_list
    # c + p
    start_time = time.time()
    a_sum = a + 5
    print("a+5: ",private_key.decrypt(a_sum))
    print(f"c+p time: {time.time() - start_time:.2f} s")

    # c - p
    start_time = time.time()
    a_sub = a - 3
    print("a-3: ",private_key.decrypt(a_sub))
    print(f"c-p time: {time.time() - start_time:.2f} s")

    # c * p
    start_time = time.time()
    b_mul = b * 5.4
    print("b*5.4: ",private_key.decrypt(b_mul))
    print(f"c*p time: {time.time() - start_time:.2f} s")

    # c / p
    start_time = time.time()
    c_div = c / 5.4
    print("c/-10.0: ",private_key.decrypt(c_div))
    print(f"c/p time: {time.time() - start_time:.2f} s")

    # c + c
    start_time = time.time()
    d_enc_add = a + b
    print("c+c: ", private_key.decrypt(d_enc_add))
    print(f"c+c time: {time.time() -start_time:.2f} s")

def main():
    paillier_test()

if __name__ == "__main__":
    main()