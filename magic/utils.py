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

import copy
import logging
import os
import shutil

from magic.cell import Cell
from magic.magic import Magic
from magic.magic_parser import MagicParser
from schematic_capture.circuit import Circuit, SubCircuit
from schematic_capture.devices import SubDevice
from schematic_capture.utils import get_bottom_up_topology, get_top_down_topology

logger = logging.getLogger(__name__)


def instantiate_circuit(Circuit: Circuit, path="magic/devices"):
    """Instantiate the devices of the given circuit, and all its possible sub-circuits in magic.

    Args:
        Circuit (Circuit): Circuit whose cell-view shall be generated.
        path (str, optional): Path where the resulting files, will be saved. Defaults to 'magic/devices'.
                            The files will be stored under:
                                <working_dir>/<path>

    """
    logger.info(f"Instantiating {Circuit} in magic.")

    # get the topology of the circuit
    topology = get_top_down_topology(Circuit)
    topology.sort(key=lambda x: x[0])

    logger.debug(f"Instantiation topology: {topology}")

    # if Devices folder exists delete it
    if os.path.exists(path):
        shutil.rmtree(path)

    # make the Devices folder
    if not os.path.exists(path):
        os.makedirs(path)

    # for each circuit instantiate the devices
    for t, c in topology:
        instantiate_devices(c, path, del_path=False)
        logger.debug(f"Instantiated devices of {c} at topological layer {t}.")


def instantiate_devices(circuit: Circuit, path="magic/devices", del_path=True):
    """Instantiate the devices of a circuit (Without the devices of possible sub-circuits).

    Args:
        Circuit (Circuit): Circuit which shall be instantiated in magic.
        path (str, optional): Path where the resulting files, will be saved. Defaults to 'magic/devices'.
                            The files will be stored under:
                                <working_dir>/<path>
        del_path (bool, optional): If the content at <path> shall be deleted, before the instantiation. Defaults to True.

    """
    logger.info(f"Instantiating devices of {circuit} in magic. Devices-path: {path}")

    # get the device generation commands
    mag = Magic(circuit)
    lines = mag.gen_devices()

    # if devices folder exists delete it
    if os.path.exists(path) and del_path:
        shutil.rmtree(path)

    # make the devices folder
    if not os.path.exists(path):
        os.makedirs(path)

    # write a tcl script to generate the devices
    file = open(path + "/init_devs.tcl", "w")
    for l_ in lines:
        file.write(l_ + "\n")
    file.close()

    # let magic generate the devices
    # check if the variable PDKPATH is set
    if "PDKPATH" in os.environ:
        # save the actual directory
        act_dir = os.getcwd()
        os.chdir(path)
        os.system(
            'magic -dnull -noconsole -rcfile ${PDKPATH}/libs.tech/magic/sky130A.magicrc "init_devs.tcl" > /dev/null'
        )
        os.chdir(act_dir)
    else:
        raise KeyError("[ERROR] Variable PDKPATH not set!")

    # if the circuit has already a cell view, update the paths
    for device in circuit.devices.values():
        if device.cell is not None:
            if isinstance(device.cell, Cell):
                device.cell.add_path(os.path.realpath(f"{path}"))


def generate_cell(name: str, path="magic/devices") -> Cell:
    """Generate a Cell-view.

    Args:
        name (str): Name of the cell/device for which the cell-view shall be generated.
        path (str, optional): Path to the magic-view of the cell. Defaults to 'magic/devices'.

    Raises:
        FileNotFoundError: If the magic-view can't be found.

    Returns:
        Cell: Generated cell-view.

    """
    logger.debug(f"Generating cell: {name}")

    if not os.path.exists(f"{path}/{name}.mag"):
        raise FileNotFoundError(f"Magic-view of cell {name} not found in {path}/!")

    # parse the magic-file
    parser = MagicParser(f"{path}/{name}.mag")

    # get the layers of the device
    layers = copy.copy(parser.layers)

    # generate the cell
    cell = Cell(name, layers)

    # add the path to the cell
    cell.add_path(os.path.realpath(f"{path}"))
    # cell.add_path(f'..{path[5:]}/')

    logger.debug(f"Generated cell {cell}.")

    return cell


def add_cells(circ: Circuit, path="magic/devices"):
    """Add a cell-view to the circuit.

    Args:
        circ (Circuit): Circuit whose cell-view shall be generated.
        path (str, optional): Path to the magic-view of the devices. Defaults to 'magic/devices'.

    """
    try:
        topology = get_top_down_topology(circ)
        topology.sort(key=lambda x: x[0])

        for _, circuit_ in topology:
            for d_name, d in circuit_.devices.items():
                if type(d) is not SubDevice:
                    cell_path = path
                    cell = generate_cell(d_name, cell_path)
                    d.set_cell(cell)
    except FileNotFoundError:
        print("Magic-view can't be found!")
        print(f"Generating new view under '{path}'!")
        instantiate_circuit(circ, path)
        add_cells(circ=circ, path=path)
    except Exception as e:
        raise ValueError(f"Adding cells to {circ} failed!") from e


def place_circuit(
    name: str, circuit: Circuit, path="magic/placement", debug=False, clean_path=True
):
    """Place the devices of circuit <Circuit> in magic.

    Args:
        name (str): Name of the top-cell.
        circuit (Circuit): Circuit which shall be placed.
        path (str, optional): Path to the resulting top-cell. Defaults to 'magic/placement'.
        debug (bool, optional): If True, only the tcl script will be generated, but not executed. Defaults to False.
        clean_path (bool, optional): If True, the content at <path> will be deleted, before stating the placement. Defaults to True.

    """
    # generate the commands to place the circuit
    mag = Magic(circuit)
    lines = mag.place_circuit(name, path="")

    # if Placement folder exists delete it
    if os.path.exists(path) and clean_path:
        shutil.rmtree(path)

    # make the Placement folder
    if not os.path.exists(path):
        os.makedirs(path)

    # write the tcl script to generate the Placement
    file = open(path + "/place_devs.tcl", "w")
    for l in lines:
        file.write(l + "\n")
    file.close()

    if not debug:
        # check if the variable PDKPATH is set
        if "PDKPATH" in os.environ:
            # let magic generate the devices
            act_dir = os.getcwd()
            os.chdir(path)
            os.system(
                'magic -dnull -noconsole -rcfile ${PDKPATH}/libs.tech/magic/sky130A.magicrc "place_devs.tcl" > /dev/null'
            )
            # delete the tcl script
            os.remove("place_devs.tcl")
            os.chdir(act_dir)
        else:
            raise KeyError("[ERROR] Variable PDKPATH not set!")


def place_circuit_hierachical(
    name: str, circuit: Circuit, path="magic/placement", clean_path=True
):
    """Do the placement of a circuit hierarchical.

        WARNING: Hierarchical placement can lead to errors, since
                 Magic scales cells "spontaneously".

    Args:
        name (str): Name of the top-cell.
        circuit (Circuit): Circuit which shall be placed.
        path (str, optional): Path of the placement. Defaults to "magic/placement".
        clean_path (bool, optional): True, if the path should be cleaned before placing the devices. Defaults to True.

    """
    # if Placement folder exists delete it
    if os.path.exists(path) and clean_path:
        shutil.rmtree(path)

    # make the Placement folder
    if not os.path.exists(path):
        os.makedirs(path)

    topology = get_bottom_up_topology(circuit)

    # sort the topology in descent order (starting with the lowest)
    topology.sort(key=lambda x: x[0], reverse=True)

    for _, circuit_ in topology:
        if isinstance(circuit, SubCircuit):
            # get the subdevice
            circ_c = copy.deepcopy(circuit_)
            sub_device = circ_c.sub_device
            macro_cell = sub_device.cell
            # center the macro-cell
            macro_cell.move_center((0, 0))
            macro_cell.rotate_center(-macro_cell.rotation)
            macro_cell._move_cells_to_bound()
            # place the subcircuit
            place_circuit(sub_device.name, circ_c, path=path, clean_path=False)

            circuit_.sub_device.cell.add_path(os.path.realpath(f"{path}"))
        else:
            place_circuit(name, circuit_, path=path, clean_path=False)
