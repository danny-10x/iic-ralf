# ========================================================================
#
#   Script to manually place a circuit.
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

import faulthandler

import matplotlib.pyplot as plt
from prettytable import PrettyTable

from environment.cell_sliding import cell_slide3
from environment.rudy import Rudy
from Magic.utils import add_cells, instantiate_circuit
from PDK.PDK import global_pdk
from Routing_v2.utils import get_nets_and_pins
from SchematicCapture.Devices import SubDevice
from SchematicCapture.Net import SubNet
from SchematicCapture.RString import include_RStrings_hierarchical
from SchematicCapture.utils import (
    Circuit,
    include_primitives_hierarchical,
    setup_circuit,
)

faulthandler.enable()

# global variables to control the placement
CIRCUIT_FILE = "Circuits/Examples/DiffAmp.spice"  # Input spice-netlist
CIRCUIT_NAME = "DiffAmp"  # Name of the top-circuit

placement = {
    "XDP_XM1_XM2": [726, 725, 270],
    "XM5": [846, 725, 90],
    "XDL_XM3_XM4": [726, 725, 270],
}

# setup the circuit
circuit = setup_circuit(CIRCUIT_FILE, CIRCUIT_NAME, [], net_rules_file=None)

# include primitive compositions into the circuit
include_primitives_hierarchical(circuit)
include_RStrings_hierarchical(circuit)

# instantiate the circuit cells in magic
instantiate_circuit(circuit, "Magic/Devices")

# add the cells to the devices
add_cells(circuit, "Magic/Devices")

# place the devices
cells = []
for dev_name, action in placement.items():
    circuit.devices[dev_name].cell.place((action[0], action[1]), action[2])
    cells.insert(0, circuit.devices[dev_name].cell)

# place cells in magic
# place_circuit('DiffAmp_non_slid', C)

# plot the placement
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot()
for d in circuit.devices.values():
    d.cell.plot(ax)

plt.show()

# slide the cells
cell_slide3(cells)

# place cells in magic
# place_circuit('DiffAmp_slid', C)

# plot the slid placement
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot()
for d in circuit.devices.values():
    d.cell.plot(ax)

plt.show()

# Calc. statistics ##################
# get all nets of the circuit - including subnets
nets_and_pins = get_nets_and_pins(circuit)

# setup RUDY for congestion estimation
rudy = Rudy(global_pdk)

# setup a dict to store the HPWLs
HPWL = {}

# get the HPWLs
for net in nets_and_pins.keys():
    rudy.add_net(net)

    if isinstance(net, SubNet):
        net_name = net.name + "_" + net.parent_device.name_suffix
    else:
        net_name = net.name

    HPWL[net_name] = net.HPWL()

# get the congestion
congestion = round(rudy.congestion(), 2)

# calculate the bounding box of the placement
bounding_box = [float("inf"), float("inf"), -float("inf"), -float("inf")]
for device in circuit.devices.values():
    device_bound = device.cell.get_bounding_box()
    bounding_box[0] = min(bounding_box[0], device_bound[0])
    bounding_box[1] = min(bounding_box[1], device_bound[1])
    bounding_box[2] = max(bounding_box[2], device_bound[2])
    bounding_box[3] = max(bounding_box[3], device_bound[3])

w = round((bounding_box[2] - bounding_box[0]) * global_pdk.scale_factor / 1e3, 2)
h = round((bounding_box[3] - bounding_box[1]) * global_pdk.scale_factor / 1e3, 2)


def _get_number_of_cells(circuit: Circuit):
    # get the total number of cells of an circuit
    n_cells = 0
    # iterate over the devices
    for device in circuit.devices.values():
        if isinstance(device, SubDevice):
            # if device is a SubDevice -> find the number of cells for the subcircuit
            subcircuit = device._circuit
            n_cells += _get_number_of_cells(subcircuit)
        else:
            # device is primitive -> has a cell
            n_cells += 1

    return n_cells


# calculate the total HPWL and setup a table
table = PrettyTable(["Parameter", "Value", "Unit"])
total_hpwl = 0
for net, hpwl in HPWL.items():
    table.add_row([
        f"HPWL({net})",
        round(hpwl * global_pdk.scale_factor / 1e3, 2),
        "um",
    ])
    total_hpwl += hpwl

# add the data
table.add_row([
    "Total HPWL",
    round(total_hpwl * global_pdk.scale_factor / 1e3, 2),
    "um",
])
table.add_row(["Congestion", congestion, "-"])
table.add_row(["Width", w, "um"])
table.add_row(["Height", h, "um"])
table.add_row(["Area", round(w * h, 2), "um2"])
table.add_row(["#Cells", _get_number_of_cells(circuit), "-"])

print(table)
