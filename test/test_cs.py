import numpy as np
import tenseal as ts

class Client:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.pub_ctx = self._public_ctx()

    def __init__(self) -> None:
        self.ctx = self._get_context()
        self.pub_ctx = self._public_ctx()

    def _get_context(self) -> ts.Context:
        # TODO: add ContextManager logic
        ...

    def _public_ctx(self) -> bytes:
        ctx_pub = self.ctx.copy()
        ctx_pub.make_context_public()
        if ctx_pub.is_private():
            raise ValueError("public context should not have private keys")
        return ctx_pub.serialize()
    
    @staticmethod
    def ndarray_type(arr: np.ndarray) -> str:
        if arr.ndim == 1:
            return "vector"
        elif arr.ndim >= 2:
            return "tensor"
        else:
            raise ValueError("Invalid input array")
    
    def send_message(self, message):
        if self.ndarray_type(message) == "vector":
            return self.pub_ctx + self.encrypt_vector(message)
        else:
            return self.pub_ctx + self.encrypt_tensor(message)
        
    def recieve_message(self, message):
        if self.ndarray_type(message) == "vector":
            return self.decrypt_vector(message)
        else:
            return self.decrypt_tensor(message)

    def encrypt_vector(self, x: np.ndarray) -> bytes:
        pub_ctx = self._public_ctx(self.ctx)
        return ts.ckks_vector(pub_ctx, x).serialize()
    
    def encrypt_tensor(self, x: np.ndarray) -> bytes:
        pub_ctx = self._public_ctx(self.ctx)
        return ts.ckks_tensor(pub_ctx, x).serialize()
    
    def decrypt_vector(self, ct: bytes) -> np.ndarray:
        return ts.ckks_vector_from(self.ctx, ct).decrypt()
    
    def decrypt_tensor(self, ct: bytes) -> np.ndarray:
        return ts.ckks_tensor_from(self.ctx, ct).decrypt()
    
class Server:
    def __init__(self, ctx):
        self.pub_ctx = None

    def get_pub_ctx(self) -> bytes:
        return self.pub_ctx
    
    def recieve_message(self, message):
        if Client.ndarray_type(message) == "vector":
            return self.get_vector(message)
        else:
            return self.get_tensor(message)

    def get_vector(self, ct: bytes) -> ts.CKKSVector:
        return ts.ckks_vector_from(ts.context_from(self.ctx), ct)
    
    def get_tensor(self, ct: bytes) -> ts.CKKSTensor:
        return ts.ckks_tensor_from(ts.context_from(self.ctx), ct)
    
def test_end2end_tenseal_simulation():
    a = np.arange(10)
    b = np.arange(10)

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2**40

    client = Client(ctx)
    send_content = client.encrypt_vector(a)

    

def test_tenseal_simulation():
    # init tenseal context
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2**40

    # simulate input
    a = [np.random.random() for _ in range(5)]
    b = np.random.random()
    c = np.random.random()

    # generate public context for encryption
    ctx_pub = ctx.copy()
    ctx_pub.make_context_public()
    assert (ctx_pub.is_public(), True)
    assert (ctx_pub.is_private(), False)  # no private keys

    # encryption with public ctx that does not have private key
    ct_a = ts.ckks_vector(ctx_pub, a)
    ct_a_bytes = ct_a.serialize()
    ctx_pub_bytes = ctx_pub.serialize()
    # send `ct_a_bytes` to server
    # send `ctx_pub_bytes` to server

    # server operation: the server need the public ctx and ct of a
    ct_a_server = ts.ckks_vector_from(ts.context_from(ctx_pub_bytes), ct_a_bytes)
    ct_res = (ct_a_server * b) * (ct_a_server * c)
    
    # ct_res.decrypt()  # cannot decrypt at public domain
    ct_res_bytes = ct_res.serialize()  # send to client

    # client decryption
    # using ctx that includes private keys to deserialize ct
    ct_res_client = ts.ckks_vector_from(ctx, ct_res_bytes)
    res = ct_res_client.decrypt()
    expect_res = (np.array(a) * b) * (np.array(a) * c)
    print(res)
    print(expect_res)
    assert (res - expect_res < 1e-6).all()

if __name__ == "__main__":
    test_end2end_tenseal_simulation()