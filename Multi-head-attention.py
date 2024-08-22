
import math
from typing import Optional,List

import torch
from torch import nn

from labml import tracker

class PrePareForMultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,heads:int,d_k:int,bias:bool):
        super().__init__()
        self.linear =nn.Linear(d_model,heads*d_k,bias=bias)
        self.heads=heads
        self.d_k=d_k
    def forward(self,x:torch.Tensor):
        head_shape=x.shape[:-1]

        x=self.linear(x)
        x=x.view(*head_shape,self.heads,self.d_k)

        return x
class MultiHeadAttention(nn.Module):
    def __init__(self,heads:int,d_model:int,dropout_prob:float=0.1,bias:bool=True):

        super().__init__()

        self.d_k=d_model//heads
        self.heads=heads

        self.query=PrePareForMultiHeadAttention(d_model,heads,self.d_k,bias=bias)
        self.key=PrePareForMultiHeadAttention(d_model,heads,self.d_k,bias=bias)
        self.value=PrePareForMultiHeadAttention(d_model,heads,self.d_k,bias=True)



        self.softmax=nn.Softmax(dim=1)

        self.output=nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout_prob)
