# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn as nn


def get_cell(cell_type):
    try:
        Cell = getattr(nn, cell_type.upper())
    except Exception:
        raise ValueError(f"Unknown RNN cell type {cell_type}")
    return Cell
