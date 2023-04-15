import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts
import unittest

import test_helper
from thex import (
    logger,
    cxt_man,
    utils,
)
from thex.xnn.attention import Attention


class TestAttention(unittest.TestCase):
    