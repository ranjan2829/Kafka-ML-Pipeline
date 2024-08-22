
import math
from typing import Optional,List

import torch
from torch import nn

from labml import tracker

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,heads:int,d_k:int,bias:bool):
        super().__init__()
