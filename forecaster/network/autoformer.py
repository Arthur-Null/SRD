from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from forecaster.network import NETWORKS

from ..module.autoformer import (
    AutoCorrelation,
    AutoCorrelationLayer,
    DataEmbedding_wo_pos,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)


@NETWORKS.register_module()
class Autoformer(nn.Module):
    """
    Implementation from https://github.com/thuml/Autoformer
    Use Informer Model
    """

    def __init__(
        self,
        max_length: Optional[int] = None,
        out_len: Optional[int] = None,
        label_len: int = 10,
        moving_avg: int = 6,
        d_model: int = 128,
        freq: str = "h",
        embed: str = "timeF",
        dropout: float = 0.1,
        factor: float = 1.0,
        n_heads: int = 4,
        d_ff: int = 512,
        activation: str = "relu",
        d_layers: int = 3,
        e_layers: int = 3,
        output_attention: bool = False,
        input_size: int = 1,
        weight_file: Optional[Path] = None,
    ):
        super(Autoformer, self).__init__()
        self.seq_len = max_length
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention
        c_out = input_size

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(input_size, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(input_size, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len :, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len :, :], x_mark_dec], dim=1)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        out = dec_out[:, -self.pred_len :, :]
        return out
