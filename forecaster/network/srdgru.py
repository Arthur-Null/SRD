from pathlib import Path
from typing import Optional

from .base import NETWORKS
import torch
import torch.nn as nn
from utilsd import use_cuda

from ..module import PositionEmbedding, get_cell
from ..module.basic import SelectItem
from ..module.srdgru import GraphRNNLayer
from ..module.srdtcn import (
    dyna_graph_constructor,
    dyna_sparse_graph,
    gdistance_fro,
    graph_constructor,
    sparse_graph,
)


@NETWORKS.register_module()
class SRDGRU(nn.Module):
    def __init__(
        self,
        spatial_embed: bool,
        temporal_embed: bool,
        temporal_emb_type: str,
        hidden_size: int,
        dropout: float,
        cell_type: str = "GRU",
        propalpha: float = 0.05,
        gcn_depth: int = 2,
        gcn: bool = True,
        dyna: bool = False,
        num_layers: int = 1,
        is_bidir: bool = True,
        max_length: int = 100,
        subgraph_size: int = 20,
        node_dim: int = 40,
        tanhalpha: int = 3,
        out_size: int = 4,
        distance: str = "fro",
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
    ):
        super().__init__()
        device = torch.device("cuda:0" if use_cuda() else "cpu")
        Cell = get_cell(cell_type)
        subgraph_size = min(subgraph_size, input_size)
        self.gc = graph_constructor(input_size, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=None)
        self.dyna_gc = dyna_graph_constructor(input_size, subgraph_size, node_dim, device, alpha=tanhalpha)
        self.graph_encoder = nn.Sequential(
            nn.Linear(1, node_dim), nn.LeakyReLU(), nn.GRU(node_dim, node_dim, batch_first=True)
        )
        self.encoder = nn.Sequential(nn.Linear(1, hidden_size), nn.LeakyReLU())
        self.regressive_encoder_static = nn.ModuleList()
        self.regressive_encoder_dynamic = nn.ModuleList()
        for _ in range(num_layers):
            # Temporal Module
            rnn = nn.Sequential(
                Cell(hidden_size, hidden_size, batch_first=True, bidirectional=is_bidir),
                SelectItem(0),
                nn.Linear(hidden_size * 2 if is_bidir else hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Spatial Module
            self.regressive_encoder_dynamic.append(GraphRNNLayer(rnn, hidden_size, input_size, max_length, True, gcn_depth, dropout, propalpha))  # type: ignore
            self.regressive_encoder_static.append(GraphRNNLayer(rnn, hidden_size, input_size, max_length, False, gcn_depth, dropout, propalpha))  # type: ignore
        self.out_layer = nn.Sequential(nn.Linear(hidden_size, out_size), nn.LeakyReLU())
        self.out_layer_dyna = nn.Sequential(nn.Linear(hidden_size, out_size), nn.LeakyReLU())

        if spatial_embed:
            self.spatial_embedding = nn.Embedding(input_size, hidden_size)  # type: ignore
        else:
            self.spatial_embedding = None
        if temporal_embed:
            self.temporal_embedding = PositionEmbedding(temporal_emb_type, hidden_size, max_length)
        else:
            self.temporal_embedding = None

        self.gdistance = gdistance_fro

        self.__output_size = input_size * out_size
        self.__hidden_size = hidden_size
        self.is_bidir = is_bidir
        self.gcn = gcn
        self.dyna = dyna
        self.idx = torch.arange(input_size).to(device)  # type: ignore
        self.subgraph_size = subgraph_size
        self.num_layers = num_layers

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size

    def get_adp(self, inputs, full=False):
        bs, temporal_dim, spatial_dim = inputs.size()

        static_adp = self.gc.fullA(self.idx)
        dyna_input = inputs.unsqueeze(1).transpose(2, 3)
        emb = self.graph_encoder(dyna_input.reshape(-1, temporal_dim, 1))[0][:, -1, :].reshape(bs, spatial_dim, -1)
        dyna_adp = self.dyna_gc.fullA(self.idx, emb)
        if full:
            return dyna_adp, static_adp
        else:
            dyna_adp = dyna_sparse_graph(dyna_adp, self.idx, self.subgraph_size)
            adp = sparse_graph(static_adp, self.idx, self.subgraph_size)
            return dyna_adp, adp

    def forward(self, inputs):
        bs, temporal_dim, spatial_dim = inputs.size()

        # Graph Learning
        static_adp = self.gc.fullA(self.idx)
        dyna_input = inputs.unsqueeze(1).transpose(2, 3)
        emb = self.graph_encoder(dyna_input.reshape(-1, temporal_dim, 1))[0][:, -1, :].reshape(bs, spatial_dim, -1)
        dyna_adp = self.dyna_gc.fullA(self.idx, emb)

        # MinmaxLoss Calculation
        max_loss = -self.gdistance(dyna_adp, static_adp.detach().unsqueeze(0).expand_as(dyna_adp))
        min_loss = self.gdistance(dyna_adp.detach(), static_adp.unsqueeze(0).expand_as(dyna_adp))
        self.minmax_loss = (min_loss, max_loss)
        self.minmax_loss_value = min_loss

        # Top-k
        dyna_adp = dyna_sparse_graph(dyna_adp, self.idx, self.subgraph_size)
        adp = sparse_graph(static_adp, self.idx, self.subgraph_size)

        inputs = self.encoder(inputs.view(bs, -1, 1))  # bs, temporal_dim * spatial_dim, 1
        if self.spatial_embedding is not None:
            spatial_embedding = self.spatial_embedding(
                torch.arange(end=spatial_dim, device=inputs.device)
            )  # spatial_dim, emb_dim
            spatial_embedding = spatial_embedding.repeat(bs, temporal_dim, 1)
            inputs = inputs + spatial_embedding
        if self.temporal_embedding is not None:
            temporal_embedding = self.temporal_embedding(
                torch.zeros(bs, temporal_dim, self.hidden_size, device=inputs.device)
            ).repeat_interleave(spatial_dim, dim=1)
            inputs = inputs + temporal_embedding
        z = inputs.reshape(bs, temporal_dim, spatial_dim, self.hidden_size).transpose(1, 2)

        # Spatial and Temporal Modules
        outs_static = outs_dyna = z
        for i in range(self.num_layers):
            outs_static = self.regressive_encoder_static[i](outs_static, adp)
            outs_dyna = self.regressive_encoder_dynamic[i](outs_dyna, dyna_adp)

        outs_static = (
            self.out_layer(outs_static.reshape(bs, spatial_dim, temporal_dim, -1))
            .transpose(2, 3)
            .reshape(bs, -1, temporal_dim)
        )
        outs_dyna = (
            self.out_layer_dyna(outs_dyna.reshape(bs, spatial_dim, temporal_dim, -1))
            .transpose(2, 3)
            .reshape(bs, -1, temporal_dim)
        )
        return outs_static, outs_static[:, :, -1], outs_dyna, outs_dyna[:, :, -1]
