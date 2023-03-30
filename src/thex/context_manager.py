import sys
from typing import Tuple
import math
import tenseal as ts


class ContextManager:
    def __init__(self, poly_mod=32768, inner_primes=35, precision_integer=5):
        '''
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
        '''
        self._table = {
            1024: 27,
            2048: 54,
            4096: 109,
            8192: 218,
            16384: 438,
            32768: 881,
        }
        self.poly_mod = poly_mod
        # larger size, more accurate, less size of multiplication depth
        self.inner_primes = inner_primes
        # bit precision before decimal point
        self.precision_integer = precision_integer
        # bit precision after decimal point
        self.precision_fractional = self.inner_primes - self.precision_integer
        # depth
        self.max_depth = math.floor(
            (self._table[self.poly_mod] - (self.precision_integer + self.size_inner_primes) * 2)
            / self.inner_primes
        )
        self.depth = 0
        # SEAL context
        self.scale = 2 ** self.inner_primes
        self.coeff_mod_bit_sizes = (
            [self.precision_integer + self.inner_primes]
            + [self.inner_primes for _ in range(self.depth)]
            + [self.precision_integer + self.inner_primes]
        )

    def get_context(self) -> Tuple[ts.Context, int]:
        """Get a tenseal context for the given parameters.
        Raises:
            ValueError: If the parameters are not valid.
        Returns:
            Tuple[ts.Context, int]: The tenseal context and the precision.
        """
        # SEAL context
        ctx = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_mod,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        ctx.global_scale = self.scale
        return ctx, int(self.poly_mod / 4)

    def depth_check(self, depth_increment):
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
                if self.depth >= self.max_depth:
                    raise ValueError("Max depth exceeded")
                self.depth += depth_increment
                try:
                    return func(*args, **kwargs) * depth_increment
                finally:
                    self.depth -= depth_increment if not sys.exc_info()[0] else 0
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
            return self.depth_check(depth_increment)(func)
        return decorator