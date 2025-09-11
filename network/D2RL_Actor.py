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
from torch_geometric.nn import MeanAggregation, SAGEConv
from torch_geometric.nn.norm import BatchNorm


class D2RLActor(nn.Module):
    """Class to define a D2RL Actor.

    Actor network for D2RL, incorporating a Graph Convolutional Network (GCN)
    and a Multi-Layer Perceptron (MLP) with D2RL-style skip connections.
    """

    def __init__(self, out_dim_x, out_dim_y):
        """Initialize actor network."""
        super().__init__()

        # Graph Convolutional Network layers
        self.conv1 = SAGEConv(in_channels=-1, out_channels=16)
        self.norm1 = BatchNorm(in_channels=16)
        self.conv2 = SAGEConv(in_channels=16, out_channels=16)
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
        batch = data.batch

        # GCN forward pass
        x = nn.functional.relu(self.conv1(x, edge_index))
        x = self.norm1(x)
        x = nn.functional.relu(self.conv2(x, edge_index))

        # Global aggregation
        x_encoded = self.mean_aggr(x, batch)

        # D2RL MLP forward pass
        x = self.norm_lin1(x_encoded)
        x = nn.functional.relu(self.lin_1(x))

        x_combined = torch.cat([x, x_encoded], dim=1)
        x = self.norm_lin2(x_combined)
        x = nn.functional.relu(self.lin_2(x))

        x_combined = torch.cat([x, x_encoded], dim=1)
        x = self.norm_lin3(x_combined)
        x = nn.functional.relu(self.lin_3(x))

        y = self.linear_y(x)
        xx = self.linear_x(x)

        # Output probabilities
        xx = nn.functional.softmax(self.linear_x(x), dim=1)
        y = nn.functional.softmax(self.linear_y(x), dim=1)
        rot = nn.functional.softmax(self.linear_rot(x), dim=1)

        return xx, y, rot
