from typing import Tuple
import logging
import math
import tenseal as ts


def get_tenseal_context(poly_mod=32768, inner_primes=35, precision_integer=5) -> Tuple[ts.Context, int]:
    """Get a tenseal context for the given parameters.

    Args:
        poly_mod (int, optional): The polynomial modulus. Defaults to 32768.
        inner_primes (int, optional): The number of inner primes. Defaults to 35.
        precision_integer (int, optional): The precision. Defaults to 5.

    Notice (from SEAL):
        8192 under 128 security level -> 218
        16384 under 128 security level -> 438
        32768 under 128 security level -> 881
    Raises:
        ValueError: If the parameters are not valid.

    Returns:
        Tuple[ts.Context, int]: The tenseal context and the precision.
    """

    _table = {
        1024: 27,
        2048: 54,
        4096: 109,
        8192: 218,
        16384: 438,
        32768: 881,
    }
    _poly_mod = poly_mod
    # need >= 30? larger size, more accurate, less size of multiplication depth
    _size_inner_primes = inner_primes
    # bit precision before decimal point
    _precision_integer = precision_integer
    # bit precision after decimal point
    _precision_fractional = _size_inner_primes - _precision_integer
    _scale = 2**_size_inner_primes
    # depth
    _depth = math.floor(
        (_table[_poly_mod] - (_precision_integer + _size_inner_primes) * 2)
        / _size_inner_primes
    )
    print("multiplication (theoretical) depth: {}".format(_depth))
    # restriction: sum(_coeff_mod_bit_sizes) < _table[_poly_mod]
    _coeff_mod_bit_sizes = (
        [_precision_integer + _size_inner_primes]
        + [_size_inner_primes for _ in range(_depth)]
        + [_precision_integer + _size_inner_primes]
    )
    # SEAL context
    ctx = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=_poly_mod,
        coeff_mod_bit_sizes=_coeff_mod_bit_sizes,
    )
    ctx.global_scale = _scale
    return ctx, int(_poly_mod / 4)

class LoggingUtils:

    def __init__(self, logger_name=None, log_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.level = log_level
        self.logger.setLevel(log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def add_file_handler(self, filename, level=logging.WARNING):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def add_console_handler(self, level=None):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level if level is None else level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
