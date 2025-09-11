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

import torch.nn as nn


class D2RLPolicy(nn.Module):
    """Class to define a d2rl actor-critic."""

    def __init__(self, actor, critic) -> None:
        """Initialize an actor-critic network."""
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, data):
        """Make a forward pass."""
        action_pred = self.actor(data)
        value_pred = self.critic(data)

        return action_pred, value_pred
