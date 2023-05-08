from typing import List

import torch as torch
import numpy as np
import tenseal as ts

from thex import cxt_man, logger, utils


def transpose(encdata: ts.CKKSTensor, perm: List[int]=None) -> ts.CKKSTensor:
    if isinstance(encdata, ts.CKKSTensor):
        return encdata.transpose(utils.get_axes_perm(encdata.shape, perm))
    else:
        logger.error(f"thex only surpport ts.CKKSTensor, but input array type is: {type(encdata)}")
        raise ValueError("Invalid input array")