# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilsd import use_cuda

from ..module.srdtcn import (
    LayerNorm,
    dilated_inception,
    dilated_inception_ablation,
    dyna_mixprop,
    dyna_sparse_graph,
    dynamic_graph_constructor,
    gdistance_fro,
    graph_constructor,
    mixprop,
    sparse_graph,
)
from .base import NETWORKS


@NETWORKS.register_module()
class SRDTCN(nn.Module):
    def __init__(
        self,
        use_inception: bool = True,
        buildA_true: bool = True,
        gcn_depth: int = 2,
        predefined_A: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 1,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        in_dim: int = 1,
        out_dim: int = 1,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: int = 3,
        skip: bool = True,
        residual: bool = True,
        gate: bool = True,
        pad: bool = True,
        layer_norm_affline: bool = True,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        nn.Module.__init__(self)
        subgraph_size = min(subgraph_size, input_size)
        self.subgraph_size = subgraph_size
        self.is_skip = skip
        self.is_residual = residual
        self.is_gate = gate
        self.is_pad = pad
        self.buildA_true = buildA_true
        self.num_nodes = input_size
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.dyna_gconv1 = nn.ModuleList()
        self.dyna_gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        device = torch.device("cuda:0" if use_cuda() else "cpu")
        self.gc = graph_constructor(
            input_size, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat
        )
        self.dyna_gc = dynamic_graph_constructor(
            input_size, subgraph_size, node_dim, device, alpha=tanhalpha, static_gc=self.gc
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(1, node_dim), nn.LeakyReLU(), nn.GRU(node_dim, node_dim, batch_first=True)
        )

        if not use_inception:
            dilated_inception_module = dilated_inception_ablation
        else:
            dilated_inception_module = dilated_inception

        self.seq_length = max_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential**j - 1) / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception_module(residual_channels, conv_channels, dilation_factor=new_dilation)
                )
                self.gate_convs.append(
                    dilated_inception_module(residual_channels, conv_channels, dilation_factor=new_dilation)
                )
                self.residual_convs.append(
                    nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1))
                )
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                self.dyna_gconv1.append(dyna_mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.dyna_gconv2.append(dyna_mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, input_size, self.seq_length - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, input_size, self.receptive_field - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.end_conv_1_dyna = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2_dyna = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True
            )

        self.idx = torch.arange(self.num_nodes).to(device)
        self.__output_size = input_size
        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    def get_adp(self, input, full=False):
        input = input.unsqueeze(1).transpose(2, 3)  # (batch_size, 1, input_size, seq_length)
        bs, _, input_size, seq_len = input.size()
        assert seq_len == self.seq_length, "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            if self.is_pad:
                input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            else:
                input = F.pad(
                    input,
                    (
                        (self.receptive_field - self.seq_length) // 2 + 1,
                        (self.receptive_field - self.seq_length) // 2,
                        0,
                        0,
                    ),
                )
        bs, _, input_size, seq_len = input.size()

        static_adp = self.gc.fullA(self.idx)
        emb = self.graph_encoder(input.reshape(-1, seq_len, 1))[0][:, -1, :].reshape(bs, input_size, -1)
        dyna_adp = self.dyna_gc.fullA(self.idx, emb)

        max_loss = -gdistance_fro(dyna_adp, static_adp.detach().unsqueeze(0).expand_as(dyna_adp))
        min_loss = gdistance_fro(dyna_adp.detach(), static_adp.unsqueeze(0).expand_as(dyna_adp))
        self.minmax_loss = (min_loss, max_loss)
        self.minmax_loss_value = min_loss
        if full:
            return dyna_adp, static_adp
        else:
            dyna_adp = dyna_sparse_graph(dyna_adp, self.idx, self.subgraph_size)
            static_adp = sparse_graph(static_adp, self.idx, self.subgraph_size)
            return dyna_adp, static_adp

    def forward(self, input, idx=None):
        input = input.unsqueeze(1).transpose(2, 3)  # (batch_size, 1, input_size, seq_length)
        bs, _, input_size, seq_len = input.size()
        assert seq_len == self.seq_length, "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            if self.is_pad:
                input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            else:
                input = F.pad(
                    input,
                    (
                        (self.receptive_field - self.seq_length) // 2 + 1,
                        (self.receptive_field - self.seq_length) // 2,
                        0,
                        0,
                    ),
                )
        bs, _, input_size, seq_len = input.size()

        # Graph Learning Layers
        static_adp = self.gc.fullA(self.idx)
        emb = self.graph_encoder(input.reshape(-1, seq_len, 1))[0][:, -1, :].reshape(bs, input_size, -1)
        dyna_adp = self.dyna_gc.fullA(self.idx, emb)

        # Calculate MinmaxLoss
        max_loss = -gdistance_fro(dyna_adp, static_adp.detach().unsqueeze(0).expand_as(dyna_adp))
        min_loss = gdistance_fro(dyna_adp.detach(), static_adp.unsqueeze(0).expand_as(dyna_adp))
        self.minmax_loss = (min_loss, max_loss)
        self.minmax_loss_value = min_loss

        # Discretize the graphs
        dyna_adp = dyna_sparse_graph(dyna_adp, self.idx, self.subgraph_size)
        static_adp = sparse_graph(static_adp, self.idx, self.subgraph_size)

        x_ori = self.start_conv(input)

        # Spatial Temporal Modules with Dynamic graph
        x = x_ori
        if self.is_skip:
            skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            if self.is_residual:
                residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            if self.is_gate:
                gate = self.gate_convs[i](x)
                gate = torch.sigmoid(gate)
                x = filter * gate
            else:
                x = filter
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            # b, c, n, t
            if self.is_skip:
                s = self.skip_convs[i](s)
                skip = s + skip
            x = self.dyna_gconv1[i](x, dyna_adp) + self.dyna_gconv2[i](x, dyna_adp.transpose(1, 2))
            # x2 = self.gconv1[i](x, static_adp) + self.gconv2[i](x, static_adp.transpose(1, 0))

            if self.is_residual:
                x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        if self.is_skip:
            skip = self.skipE(x) + skip
        else:
            skip = self.skipE(x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1_dyna(x))  # T -> T' -> T'' -> ... -> 1
        x = self.end_conv_2_dyna(x)  # B * T * I * D
        x_dyna = x

        # Spatial Temporal Modules with Prior graph
        x = x_ori
        for i in range(self.layers):
            if self.is_residual:
                residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            if self.is_gate:
                gate = self.gate_convs[i](x)
                gate = torch.sigmoid(gate)
                x = filter * gate
            else:
                x = filter
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            # b, c, n, t
            if self.is_skip:
                s = self.skip_convs[i](s)
                skip = s + skip
            x = self.gconv1[i](x, static_adp) + self.gconv2[i](x, static_adp.transpose(1, 0))

            if self.is_residual:
                x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        if self.is_skip:
            skip = self.skipE(x) + skip
        else:
            skip = self.skipE(x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))  # T -> T' -> T'' -> ... -> 1
        x = self.end_conv_2(x)  # B * T * I * D
        x_static = x

        return x_static, x_static.squeeze(-1).squeeze(1), x_dyna, x_dyna.squeeze(-1).squeeze(1)
