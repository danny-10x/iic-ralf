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


class D2RLCritic(nn.Module):
    """Class to define a d2rl critic.

    Critic network for D2RL, which uses a Graph Convolutional Network (GCN)
    to process graph data and a D2RL-style MLP to estimate the Q-value.
    """

    def __init__(self):
        """Initialize critic network."""
        super().__init__()

        # Graph Convolutional Network (GCN) layers
        self.conv1 = SAGEConv(in_channels=-1, out_channels=16)
        self.norm1 = BatchNorm(in_channels=16)
        self.conv2 = SAGEConv(in_channels=16, out_channels=16)
        self.mean_aggr = MeanAggregation()

        self.norm_lin1 = torch.nn.BatchNorm1d(16)
        self.lin_1 = torch.nn.Linear(16, 16)
        self.norm_lin2 = torch.nn.BatchNorm1d(32)
        self.lin_2 = torch.nn.Linear(32, 16)
        self.norm_lin3 = torch.nn.BatchNorm1d(32)
        self.lin_3 = torch.nn.Linear(32, 16)

        # Final output layer for the Q-value
        self.linear = nn.Linear(in_features=16, out_features=1)

    def forward(self, data):
        """Make a forward pass."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.norm1(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x_encoded = self.mean_aggr(x, batch)
        x = self.norm_lin1(x_encoded)
        x = self.lin_1(x)
        x = nn.functional.relu(x)
        x = self.norm_lin2(torch.concatenate([x, x_encoded], dim=1))
        x = self.lin_2(x)
        x = nn.functional.relu(x)
        x = self.norm_lin3(torch.concatenate([x, x_encoded], dim=1))
        x = self.lin_3(x)
        x = nn.functional.relu(x)

        x = self.linear(x)

        return x
