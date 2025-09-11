# ========================================================================
#
# SPDX-FileCopyrightText: 2023 Jakob Ratschenberger
# Johannes Kepler University, Institute for Integrated Circuits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ========================================================================

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, MeanAggregation
from torch_geometric.nn.norm import BatchNorm


class GatD2rlActor(nn.Module):
    """Class to define GAT D2RL Actor network.

    Actor network for D2RL, incorporating a Graph Attention Network (GAT)
    and a Multi-Layer Perceptron (MLP) with D2RL-style skip connections.
    """

    def __init__(self, out_dim_x, out_dim_y):
        """Initialize actor network."""
        super().__init__()

        # Graph Attention Network (GAT) layers
        self.gat1 = GATConv(in_channels=-1, out_channels=16, edge_dim=2)
        self.norm1 = BatchNorm(in_channels=16)
        self.gat2 = GATConv(in_channels=16, out_channels=16, edge_dim=2)
        self.mean_aggr = MeanAggregation()

        # D2RL-style MLP with skip connections
        self.lin_1 = nn.Linear(in_features=16, out_features=16)
        self.norm_lin1 = nn.BatchNorm1d(num_features=16)
        self.lin_2 = nn.Linear(in_features=32, out_features=16)
        self.norm_lin2 = nn.BatchNorm1d(num_features=32)
        self.lin_3 = nn.Linear(in_features=32, out_features=16)
        self.norm_lin3 = nn.BatchNorm1d(num_features=32)

        # Output layers for each action dimension
        self.linear_x = nn.Linear(in_features=16, out_features=out_dim_x)
        self.linear_y = nn.Linear(in_features=16, out_features=out_dim_y)
        self.linear_rot = nn.Linear(in_features=16, out_features=4)

    def forward(self, data):
        """Make a forward pass."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # GAT Forward pass
        x_encoded = self.graph_embedding(x, edge_index, edge_attr=edge_attr)
        x_encoded = nn.functional.relu(x_encoded)
        x_encoded = self.norm1(x_encoded)
        x_encoded = self.graph_embedding2(x_encoded, edge_index, edge_attr=edge_attr)
        x_encoded = nn.functional.relu(x_encoded)

        # Global aggregation
        x_encoded = self.mean_aggr(x_encoded, batch)

        # D2RL MLP forward pass
        x = self.norm_lin1(x_encoded)
        x = self.lin_1(x)
        x = nn.functional.relu(x)
        x = self.norm_lin2(torch.concatenate([x, x_encoded], dim=1))
        x = self.lin_2(x)
        x = nn.functional.relu(x)
        x = self.norm_lin3(torch.concatenate([x, x_encoded], dim=1))
        x = self.lin_3(x)
        x = nn.functional.relu(x)

        # Output probabilities
        y = nn.functional.softmax(self.linear_y(x), dim=1)
        xx = nn.functional.softmax(self.linear_x(x), dim=1)
        rot = nn.functional.softmax(self.linear_rot(x), dim=1)

        return xx, y, rot
