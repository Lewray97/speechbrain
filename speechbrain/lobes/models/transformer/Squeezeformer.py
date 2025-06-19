# Adapted from:
# https://github.com/upskyy/Squeezeformer/blob/main/squeezeformer

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


"""
DW Sep. Convs Subsampling
"""


class DepthwiseConv2d(nn.Module):
    """This class implements the depthwise 2d convolution.

    Only channel-wise convolution is applied to the input

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Kernel size of the convolutional filters
    stride : int
        Stride factor of the convolutional filters
    padding : int or tuple
        Zero-padding added to both sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=2,
        padding=0,
    ):
        super(DepthwiseConv2d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs):
        """Returns the output of the convolution.

        Arguments
        ---------
        inputs : torch.Tensor (batch, in_channels, time)
            input to convolve. 3d tensors are expected.
        """
        return self.conv(inputs)


class PointwiseConv2d(nn.Module):
    """This class implements the pointwise 2d convolution.

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Kernel size of the convolutional filters
    stride : int
        Stride factor of the convolutional filters
    padding : int or tuple
        Zero-padding added to both sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
    ):
        super(PointwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
        )

    def forward(self, inputs):
        """Returns the output of the convolution.

        Arguments
        ---------
        inputs : torch.Tensor (batch, in_channels, time)
            input to convolve. 3d tensors are expected.
        """
        return self.conv(inputs)


class DWSepConvSubsampling(nn.Module):
    """This class implements the depthwise separable 2d convolution subsampling.

    The input will be subsampling to 1/4 length

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(DWSepConvSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            DepthwiseConv2d(out_channels, out_channels, kernel_size=3, stride=2),
            PointwiseConv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, inputs, input_lengths):
        """Returns the output of the convolution subsampling.

        Arguments
        ---------
        inputs : torch.Tensor (batch, seq_length, dim)
            input to convolve. 3d tensors are expected.
        input_lengths : torch.Tensor (batch)
        """
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths


"""
MHA Module in Squeezeformer Block
"""


class RelPositionalEncoding(nn.Module):
    """This class implements the relative positional encoding.

    Arguments
    ---------
    d_model : int
        Embedding dimension
    max_len : int
        Maximum input length
    """

    def __init__(self, d_model, max_len=5000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        """Returns the positional embedding of input.

        Arguments
        ---------
        x : torch.Tensor (B, T, C)

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """This class implements the multi-head attention with relative positional encoding, which was
    proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Arguments
    ---------
    d_model : int
        Embedding dimension
    num_heads : int
        The number of attention heads
    dropout_p : float
        Dropout rate
    """

    def __init__(self, d_model, num_heads, dropout_p):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, pos_embedding, mask=None):
        """ 
        Arguments
        ---------- 
        query : tensor
            (batch, time, dim) containing query vector
        key : tensor
            (batch, time, dim) containing key vector
        value : tensor
            (batch, time, dim) containing value vector
        pos_embedding : tensor
            (batch, time, dim) the positional embedding tensor
        mask : tensor 
            (batch, 1, time2) or (batch, time1, time2) 
            containing indices to be masked
        
        Outputs
        ---------- 
        outputs : tensor 
        """
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


class MHAModule(nn.Module):
    """
    Arguments
    ---------
    d_model : int
        Embedding dimension
    num_heads : int
        The number of attention heads
    dropout_p : float
        Dropout rate
    """
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super(MHAModule, self).__init__()
        # Learnable Scaling Layer to replace preLN for stablizing training
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs, mask=None):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, time, dim) containing input vector
        mask : tensor
            (batch, 1, time2) or (batch, time1, time2)
            Tensor containing indices to be masked
        Outputs
        ---------
        outputs: tensor
            (batch, time, dim)
        """
        outputs = inputs * self.scale + self.bias
        
        batch_size = outputs.size(0)
        relpos_emb = self.positional_encoding(outputs)
        relpos_emb = relpos_emb.repeat(batch_size, 1, 1)
        
        outputs = self.attention(outputs, outputs, outputs, relpos_emb)
        return self.dropout(outputs)


"""
Convlution Module in Squeezeformer Block
"""


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module.
    """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    """This class implements the depthwise 1d convolution.

    Only channel-wise convolution is applied to the input

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Kernel size of the convolutional filters
    stride : int
        Stride factor of the convolutional filters
    padding : int or tuple
        Zero-padding added to both sides of the input
    bias : bool
        If True, adds a learnable bias to the output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        super(DepthwiseConv1d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, in_channels, time)

        Outputs
        ---------
        outputs : tensor
            (batch, out_channels, time)
        """
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """This class implements the poiontwise 1d convolution.

    Only point-wise convolution is applied to the input

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride factor of the convolutional filters
    padding : int or tuple
        Zero-padding added to both sides of the input
    bias : bool
        If True, adds a learnable bias to the output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        bias=True,
    ):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, in_channels, time)

        Outputs
        ---------
        outputs : tensor
            (batch, out_channels, time)
        """
        return self.conv(inputs)


class ConvModule(nn.Module):
    """This class implements the convolution module in squeezeformer blocks

    It starts with a pointwise convolution and a swish activation function.
    This is followed by a single 1-D depthwise convolution layer. 
    Batchnorm is deployed just after the convolution to aid training deep models.


    Arguments
    ---------
    in_channels : int
        Number of input channels
    kernel_size : int or tuple
        Kernel size of the convolutional filters
    dropout_p : float
        Dropout rate
    """

    def __init__(
        self,
        in_channels,
        kernel_size=31,
        expansion_factor=2,
        dropout_p=0.1,
    ):
        super(ConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        # Learnable Scaling Layer to replace preLN for stablizing training
        self.scale = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            Swish(),
            DepthwiseConv1d(in_channels * expansion_factor, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, time, dim)
        
        Outputs
        ---------
        outputs : tensor
            (batch, time, dim)
        """
        outputs = inputs * self.scale + self.bias
        return self.sequential(outputs).transpose(1, 2)


"""
Feed Forward Module in Squeezeformer Block
"""


class FeedForwardModule(nn.Module):
    """This class implements the feed forward module in sequeezeformer layers
  
    This module employes Swish activation and dropout to regularize the network.
    
    Arguments
    ---------
    encoder_dim : int
        Dimension of squeezeformer encoder
    expansion_factor : int or tuple
        Expansion rate
    dropout_p : float
        Dropout rate
    """

    def __init__(
        self,
        encoder_dim=256,
        expansion_factor=4,
        dropout_p=0.1,
    ):
        super(FeedForwardModule, self).__init__()
        # Learnable Scaling Layer to replace preLN for stablizing training
        self.scale = nn.Parameter(torch.ones(encoder_dim))
        self.bias = nn.Parameter(torch.zeros(encoder_dim))
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, time, dim)
        
        Outputs
        ---------
        outputs : tensor
            (batch, time, dim)
        """
        outputs = inputs * self.scale + self.bias
        return self.sequential(outputs)


"""
Residual Connection
"""


class ResidualConnection(nn.Module):
    """ This class implements residual connection between Module
    
    Arguments
    ---------
    module : nn.Module
    """

    def __init__(self, module):
        super(ResidualConnection, self).__init__()
        self.module = module
        
    def forward(self, inputs):
        return self.module(inputs) + inputs


"""
Squeezeformer Block
"""


class SqueezeformerBlock(nn.Module):
    """This class implements a single Squezeformer Block.
    
    It is similar to the standard Transformer encoder structure,
    where the MHA and convolution modules are directly followed 
    by a feed forward module.

    Arguments
    ---------
    encoder_dim : int
        Dimension of squeezeformer encoder
    num_attention_heads : int
        Number of attention heads
    feed_forward_expansion_factor : int
        Expansion rate of feed forward module
    conv_expansion_factor : int
        Expansion rate of convolution module
    feed_forward_dropout_p : float
        Dropout rate of feed forward module
    conv_dropout_p : float
        Dropout rate of convolution module
    conv_kernel_size : int or tuple
        Size of the convolutional kernel
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerBlock, self).__init__()
        self.sequential = nn.Sequential(
            ResidualConnection(
                module=MHAModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                )
            ),
            nn.LayerNorm(encoder_dim), # Post-LN
            ResidualConnection(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
            ),
            nn.LayerNorm(encoder_dim), # Post-LN
            ResidualConnection(
                module=ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
            ),
            nn.LayerNorm(encoder_dim), # Post-LN
            ResidualConnection(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
            ),
            nn.LayerNorm(encoder_dim), # Post-LN
        )

    def forward(self, inputs):
        """
        Arguments
        ---------
        inputs : tensor
            (batch, time, dim)
        
        Outputs
        ---------
        outputs : tensor
            (batch, time, dim)
        """
        return self.sequential(inputs)


"""
Time Resolution Reduction Layer
"""


class TimeReductionLayer(nn.Module):
    """This class implements the time reduction layer in squeezeformer. 
    
    It reduces the time resolution of input to 1/2.

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Kernel size of the convolutional filters
    stride : int
        Stride factor of the convolutional filters
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
    ):
        super(TimeReductionLayer, self).__init__()
        self.sequential = nn.Sequential(
            DepthwiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            Swish(),
        )

    def forward(self, inputs, input_lengths):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        output_lengths = input_lengths >> 1
        output_lengths -= 1
        return outputs, output_lengths


"""
Recover Time Resolution
"""


def recover_resolution(inputs):
    """This function recovers previously reduced time resolution 
    """
  
    outputs = list()

    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
    return torch.stack(outputs, dim=1)


"""
Squeezeformer
"""


class Squeezeformer(nn.Module):
    """This class implements the complete Squeezeformer encoder 
    
    It contains DW Sep. Conv subsampling layer subsampling and  
    squeezeformer blocks.

    Arguments
    ---------
    input_dim : int
        Dimension of input vector
    encoder_dim : int
        Dimension of Squeezeformer encoder
    num_layers : int
        Number of Squeezeformer encoder layers
    reduce_layer_index : int
        The layer index to reduce time resolution
    recover_layer_index : int
        The layer index to recover time resolution
    num_attention_heads : int
        Number of attention heads
    feed_forward_expansion_factor : int
        Expansion rate of feed forward module
    conv_expansion_factor : int
        Expansion rate of convolution module
    feed_forward_dropout_p : float
        Dropout rate of feed forward module
    conv_dropout_p : float
        Dropout rate of convolution module
    attention_dropout_p : float
        Dropout rate of attention modul
    conv_kernel_size : int or tuple
        Size of the convolutional kernel
    half_step_residual : bool
        Indicates whether to use half step residual
    """

    def __init__(
        self,
        input_dim=80,
        encoder_dim=144,
        num_layers=16,
        reduce_layer_index=7,
        recover_layer_index=15,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        input_dropout_p=0.1,
        feed_forward_dropout_p=0.1,
        attention_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=31,
    ):
        super(Squeezeformer, self).__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = reduce_layer_index  # Unet structure from 40ms to 80ms
        self.recover_layer_index = recover_layer_index  # recover from 80ms to 40ms
        self.conv_subsample = DWSepConvSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.time_reduction_layer = TimeReductionLayer()
        self.time_reduction_proj = nn.Linear((encoder_dim - 1) // 2, encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.recover_tensor = None

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx < reduce_layer_index:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                    )
                )
            elif reduce_layer_index <= idx < recover_layer_index:
                self.layers.append(
                    ResidualConnection(
                        module=SqueezeformerBlock(
                            encoder_dim=encoder_dim,
                            num_attention_heads=num_attention_heads,
                            feed_forward_expansion_factor=feed_forward_expansion_factor,
                            conv_expansion_factor=conv_expansion_factor,
                            feed_forward_dropout_p=feed_forward_dropout_p,
                            attention_dropout_p=attention_dropout_p,
                            conv_dropout_p=conv_dropout_p,
                            conv_kernel_size=conv_kernel_size,
                        )
                    )
                )
            else:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                    )
                )

    def forward(self, inputs, input_lengths):
        """Forward propagate a inputs for Squeezeformer training.

        Arguments
        ---------
        inputs : torch.Tensor 
            (batch, seq_length, dimension)
        input_lengths : torch.Tensor 
            (batch)
        
        Outputs
        ---------
        inputs : torch.Tensor 
            (batch, seq_length, dimension)
        input_lengths : torch.Tensor 
            (batch)
        """
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_proj(outputs)

        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                self.recover_tensor = outputs
                outputs, output_lengths = self.time_reduction_layer(
                    outputs, output_lengths
                )
                outputs = self.time_reduction_proj(outputs)

            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                length = outputs.size(1)
                outputs = self.time_recover_layer(outputs)
                outputs += self.recover_tensor[:, :length, :]
                output_lengths *= 2

            outputs = layer(outputs)
        return outputs, output_lengths
