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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schematic_capture.circuit import Circuit
    from schematic_capture.net import Net
    from schematic_capture.ports import Pin


from schematic_capture.devices import NTermDevice, SubDevice


def get_nets_and_pins(circuit: Circuit) -> dict[Net, list[Pin]]:
    """Get all pins connected to a net.

    Args:
        circuit (Circuit): Circuit for which all nets should be analyzed.

    Returns:
        dict: key: Net, to which the pins belong. value: List of devices terminals.

    """
    nets = {}
    # iterate over each device of the circuit
    for d in circuit.devices.values():
        if isinstance(d, SubDevice):
            # if the device is a sub-device
            # -> get all nets and pins of the devices sub-circuit
            sub_nets = get_nets_and_pins(d._circuit)
            for k, v in sub_nets.items():
                if k.parent_net:
                    # if the net has a parent net, add the pins to the parent net
                    add_to_dict(k.parent_net, v, nets)
                else:
                    # add pins to the net
                    add_to_dict(k, v, nets)
        else:
            assert isinstance(d, NTermDevice)
            for net in d.nets.values():
                add_to_dict(net, d.get_terminals_connected_to_net(net), nets)

    return nets


def add_to_dict(net, d, nets):
    """Add nets to dict."""
    if net in nets:
        nets[net].extend(d)
    else:
        nets[net] = d
