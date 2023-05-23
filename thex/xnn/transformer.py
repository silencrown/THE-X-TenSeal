import torch
import torch.nn as nn

from thex import logger
from thex import cxt_man

from .Module import FHELayer
from .attention import MultiHeadedAttention, EncAttention
from .ffn import PositionwiseFeedForward, EncPositionwiseFeedForward
from .layernorm import LayerNorm, EncLayerNorm, ApproxLayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class ApproxSublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(ApproxSublayerConnection, self).__init__()
        self.norm = ApproxLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncSubLayerConnection(FHELayer):

    def __init__(self, torch_nn):
        super(EncSubLayerConnection, self).__init__()
        self.norm = EncLayerNorm(torch_nn)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + sublayer(self.norm(x))

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class ApproxTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

    
class EncTransformerBlock(nn.Module):

    def __init__(self, torch_nn, hidden, attn_heads, feed_forward_hidden):

        super().__init__()
        self.attention = EncAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = EncPositionwiseFeedForward(d_model=hidden, 
                                                       d_ff=feed_forward_hidden, 
                                                       torch_nn=torch_nn.feed_forward)
        self.input_sublayer = EncSubLayerConnection(size=hidden, 
                                                    torch_nn=torch_nn.input_sublayer)
        self.output_sublayer = EncSubLayerConnection(size=hidden, 
                                                     torch_nn=torch_nn.output_sublayer)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x