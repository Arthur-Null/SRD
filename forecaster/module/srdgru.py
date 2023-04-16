# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from .srdtcn import dyna_mixprop, mixprop


class GraphRNNLayer(nn.Module):
    def __init__(
        self,
        regressive_encoder: nn.Module,
        hidden_size: int,
        input_size: int,
        max_length: int,
        dyna: int = True,
        gcn_depth: int = 2,
        dropout: float = 0.5,
        propalpha: float = 0.1,
    ):
        super(GraphRNNLayer, self).__init__()
        if dyna:
            self.gconv1 = dyna_mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
            self.gconv2 = dyna_mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
        else:
            self.gconv1 = mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
            self.gconv2 = mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)

        self.regressive_encoder = regressive_encoder
        self.norm1 = nn.LayerNorm([input_size, max_length, hidden_size])
        self.norm2 = nn.LayerNorm([hidden_size, input_size, max_length])

    def forward(self, x, adj):
        # x: [batch_size, spatial_dim, seq_len, hidden_size]
        bs, sd, sl, hs = x.shape
        x = self.regressive_encoder(self.norm1(x).reshape(bs * sd, sl, hs)) + x.reshape(bs * sd, sl, hs)
        x = x.reshape(bs, sd, sl, -1).transpose(1, 3).transpose(2, 3)  # [batch_size, hidden_size, spatial_dim, seq_len]
        norm_x = self.norm2(x)
        x = self.gconv1(norm_x, adj) + self.gconv2(norm_x, adj.transpose(-1, -2)) + x
        x = x.transpose(1, 3).transpose(1, 2)
        return x


class DoubleGraphRNNLayer(nn.Module):
    def __init__(
        self,
        regressive_encoder: nn.Module,
        hidden_size: int,
        input_size: int,
        max_length: int,
        gcn_depth: int = 2,
        dropout: float = 0.5,
        propalpha: float = 0.1,
    ):
        super(DoubleGraphRNNLayer, self).__init__()
        self.dyna_gconv1 = dyna_mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
        self.dyna_gconv2 = dyna_mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
        self.gconv1 = mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)
        self.gconv2 = mixprop(hidden_size, hidden_size, gcn_depth, dropout, propalpha)

        self.regressive_encoder = regressive_encoder
        self.norm1 = nn.LayerNorm([input_size, max_length, hidden_size])
        self.norm2 = nn.LayerNorm([hidden_size, input_size, max_length])

        self.graph_merge = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, adj, dyna_adj):
        # x: [batch_size, spatial_dim, seq_len, hidden_size]
        bs, sd, sl, hs = x.shape
        x = self.regressive_encoder(self.norm1(x).reshape(bs * sd, sl, hs)) + x.reshape(bs * sd, sl, hs)
        x = x.reshape(bs, sd, sl, -1).transpose(1, 3).transpose(2, 3)  # [batch_size, hidden_size, spatial_dim, seq_len]
        norm_x = self.norm2(x)
        x_static = (self.gconv1(norm_x, adj) + self.gconv2(norm_x, adj.transpose(-1, -2))).permute(0, 2, 3, 1)
        x_dyna = (self.dyna_gconv1(norm_x, dyna_adj) + self.dyna_gconv2(norm_x, dyna_adj.transpose(-1, -2))).permute(
            0, 2, 3, 1
        )
        x = self.graph_merge(torch.cat([x_static, x_dyna], dim=3)) + x.permute(0, 2, 3, 1)
        return x
