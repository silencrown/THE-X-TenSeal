import numpy as np
import tenseal as ts


class Server:
    def __init__(self, ctx):
        self.pub_ctx = None

    def get_pub_ctx(self) -> bytes:
        return self.pub_ctx

    def get_vector(self, ct: bytes) -> ts.CKKSVector:
        return ts.ckks_vector_from(ts.context_from(self.ctx), ct)
    
    def get_tensor(self, ct: bytes) -> ts.CKKSTensor:
        return ts.ckks_tensor_from(ts.context_from(self.ctx), ct)