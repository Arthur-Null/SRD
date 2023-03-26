import torch.nn as nn


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = "selectitem"
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
