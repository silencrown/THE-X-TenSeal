import sys
from typing import Tuple
import math

import numpy as np
import tenseal as ts
import torch

from thex import logger
from thex import utils


class ContextManager:
    # TODO: need to generate the peremeters when giving specific depth
    # def __init__(self, poly_mod=32768, inner_primes=35, precision_integer=5):
    def __init__(self, poly_mod=8192, inner_primes=21, precision_integer=20):
        self._table = {
            1024: 27,
            2048: 54,
            4096: 109,
            8192: 218,
            16384: 438,
            32768: 881,
        }
        self._max_size = 0
        self._max_depth = 0
        self._depth = 0
        self._context = None
        self._scale = 0
        self._setup_context(poly_mod, inner_primes, precision_integer)

    def __str__(self):
        return f"ContextManager(\n    max_size={self._max_size},\n    max_depth={self._max_depth},\n    " \
               f"depth={self._depth},\n    context={self._context},\n    scale={self._scale})"

    @property
    def max_size(self):
        return self._max_size

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def depth(self):
        return self._depth
    
    @property
    def context(self):
        return self._context

    def _setup_context(self, poly_mod, inner_primes, precision_integer) -> Tuple[ts.Context, int]:
        """Get a tenseal context for the given parameters.

        Args:
        - poly_mod (int, optional): The polynomial modulus. Defaults to 32768.
        - inner_primes (int, optional): The number of inner primes. Defaults to 35.
        - precision_integer (int, optional): The precision. Defaults to 5.

        Notice (from SEAL):
            8192 under 128 security level -> 218
            16384 under 128 security level -> 438
            32768 under 128 security level -> 881

        Restriction: 
            sum(_coeff_mod_bit_sizes) < _table[_poly_mod]

        Raises:
            ValueError: If the parameters are not valid.

        Returns:
            Tuple[ts.Context, int]: The tenseal context and the precision.
        """

        self._poly_mod = poly_mod
        # larger size, more accurate, less size of multiplication depth
        self._inner_primes = inner_primes
        # bit precision before decimal point
        self._precision_integer = precision_integer
        # bit precision after decimal point
        self._precision_fractional = self._inner_primes - self._precision_integer
        self._scale = 2 ** self._inner_primes
        # depth
        self._max_depth = math.floor(
            (self._table[self._poly_mod] - (self._precision_integer + self._inner_primes) * 2)
            / self._inner_primes
        )
        # current depth
        self._depth = 0
        # coeff_mod_bit_sizes
        self._coeff_mod_bit_sizes = (
            [self._precision_integer + self._inner_primes]
            + [self._inner_primes for _ in range(self._max_depth)]
            + [self._precision_integer + self._inner_primes]
        )
        # SEAL context
        ctx = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._poly_mod,
            coeff_mod_bit_sizes=self._coeff_mod_bit_sizes,
        )
        ctx.global_scale = self._scale
        ctx.generate_galois_keys()
        # ctx.generate_relin_keys()
        self._max_size = int(self._poly_mod / 4)
        self._context = ctx

    def _depth_check(self, depth_increment):
        """
        Decorator that limits the depth of a function.

        Args:
        - depth_increment (int): The amount to increment the depth counter by when entering the decorated function.

        Returns:
        - A new function that wraps the original function, and enforces the depth limit.

        Raises:
        - ValueError: If the maximum depth is exceeded.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self._depth + depth_increment >= self._max_depth:
                    raise ValueError("Max depth exceeded. max_depth: {self._max_depth}, depth: {self._depth}, depth increment: {depth_increment}")
                try:
                    return func(*args, **kwargs)
                finally:
                    self._depth += depth_increment if not sys.exc_info()[0] else 0
                    logger.debug(f"depth increased to: {self._depth}")
            return wrapper
        return decorator
    
    def depth_limiter(self, depth_increment=1):
        """
        Creates a decorator that limits the depth of a function.

        Args:
        - depth_increment (int): The amount to increment the depth counter by when entering the decorated function.

        Returns:
        - A decorator function that can be used to decorate other functions.

        Example:
            context_manager = ContextManager()

            @context_manager.depth_limiter(depth_increment=2)
            def my_function(x):
                return x ** 3

            result = my_function(2)
        """
        def decorator(func):
            return self._depth_check(depth_increment)(func)
        return decorator

    def depth_refresher(self):
        """
        Creates a decorator that updates the depth of a function.

        Returns:
        - A decorator function that can be used to decorate other functions.

        Example:
            context_manager = ContextManager()

            @context_manager.depth_renew()
            def my_function(x):
                return relu(x)

            result = my_function(list_of_inputs)
        """
        def decorator(func):
            def warpper(*args, **kwargs):
                self._depth = 0
                return func(*args, **kwargs)
            return warpper
        return decorator

    def encrypt(self, data):
        """
        Encrypt data.

        Args:
        - data (np.ndarray or list): The data to encrypt.

        Returns:
        - ts.CKKSVector: The encrypted data.
        - ts.CKKSTensor: The encrypted data.
        """
        # check datatype
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass
        elif torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        else:
            raise ValueError(f"The data type {type(data)} is not supported to encryption.")
        
        str = utils.ndarray_type(data)

        if str == "tensor":
            return ts.ckks_tensor(self._context, data.tolist())
        elif str == "vector":
            return ts.ckks_vector(self._context, data.tolist())
        else:
            raise ValueError(f"The data size {data.shape} is not supported to encryption.")

    def decrypt(self, data) -> list:
        """
        Decrypt data.

        Args:
        - data (ts.CKKSVector or ts.CKKSTensor): The data to decrypt.

        Returns:
        - list: The decrypted data.
        """

        if isinstance(data, ts.CKKSVector):
            return data.decrypt()
        elif isinstance(data, ts.CKKSTensor):
            plain_tensor = data.decrypt()  # return a ts.PlainTensor object
            return plain_tensor.tolist()
        else:
            raise ValueError(f"The data type {type(data)} is not a Tenseal supported obj to decryption.")

cxt_man = ContextManager()