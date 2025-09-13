# ========================================================================
#
#   Script to generate the placement of a circuit, by using reinforcement learning.
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
import logging
import pickle
from logging.handlers import RotatingFileHandler

from environment.utils import do_bottom_up_placement
from magic.magic_die import MagicDie
from magic.utils import add_cells, instantiate_circuit
from schematic_capture.rstring import include_rstrings_hierarchical
from schematic_capture.utils import include_primitives_hierarchical, setup_circuit

faulthandler.enable()

#########################################################################

# global variables to control the placement
CIRCUIT_FILE = "circuits/examples/DiffAmp.spice"  # Input spice-netlist
CIRCUIT_NAME = "DiffAmp"  # Name of the top-circuit
NET_RULES_FILE = "net_rules/net_rules_DiffAmp.json"  # Net-rules definition file
N_PLACEMENTS = 1000  # Number of trial placements per circuit/subcircuit

USE_LOGGER = False  # If True, debug information will be logged under "logs/{CIRCUIT_NAME}_placement.log".
INSTANTIATE_CELLS_IN_MAGIC = (
    True  # If True, the devices cell-view will be instantiated in Magic
)
N_PLACEMENTS_PER_ROLLOUT = 100  # Number of trial placements per RL - rollout
DEF_FILE = None  # Def file of the circuit
SHOW_STATS = True  # Show statistics of the placement

#########################################################################


def rl_placement():
    """Run the RL-based placement for a given circuit."""
    if USE_LOGGER:
        # Setup a logger
        log_handler = RotatingFileHandler(
            filename=f"logs/{CIRCUIT_NAME}_placement.log",
            mode="w",
            maxBytes=100e3,
            backupCount=1,
            encoding="utf-8",
        )
        log_handler.setLevel(logging.DEBUG)
        logging.basicConfig(
            handlers=[log_handler],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        )

    # setup the circuit
    circuit = setup_circuit(
        CIRCUIT_FILE, CIRCUIT_NAME, [], net_rules_file=NET_RULES_FILE
    )

    # include primitive compositions into the circuit
    include_primitives_hierarchical(circuit)
    include_rstrings_hierarchical(circuit)

    # instantiate the circuit cells in magic
    if INSTANTIATE_CELLS_IN_MAGIC:
        instantiate_circuit(circuit, "Magic/Devices")

    # add the cells to the devices
    add_cells(circuit, "magic/devices")

    # define a die for the circuit
    die = MagicDie(circuit=circuit, def_file=DEF_FILE)

    # do the placement by training a RL-agent
    do_bottom_up_placement(
        circuit,
        N_PLACEMENTS,
        N_PLACEMENTS_PER_ROLLOUT,
        use_weights=False,
        show_stats=SHOW_STATS,
    )

    # save the placed circuit
    file = open(f"placement_circuits/{CIRCUIT_NAME}_placement.pkl", "wb")
    pickle.dump(die, file)
    file.close()


if __name__ == "__main__":
    rl_placement()
