import numpy as np
import tenseal as ts

from thex import logger
from thex.service.Xend import Xend


class Client(Xend):
    def __init__(self, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.pub_ctx = self._public_ctx()

    def __init__(self) -> None:
        self.ctx = self._get_context()
        self.pub_ctx = self._public_ctx()

    def _get_context(self) -> ts.Context:
        # TODO: get context from ContextManager
        pass

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
        elif self.ndarray_type(message) == "tensor":
            return self.pub_ctx + self.encrypt_tensor(message)
        else:
            raise ValueError(f"Invalid input array {message.dtype}")
        
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
    
