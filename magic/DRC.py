# ========================================================================
#
# Collection of methods to perform a DRC.
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
    from magic.cell import Cell

import os
import shutil
import subprocess


def drc_collidates(cell: Cell, cells: list[Cell]) -> bool:
    """Check if <cell> collidates with one of <cells>.

    Args:
        cell (Cell): Cell which gets checked.
        cells (list(Cell)): List of cells.

    Returns:
        bool: True if cell collidates with one of cells.

    """
    for c in cells:
        if cell.collidates(c):
            return True

    return False


def drc_collidates_all(cells: list[Cell]) -> bool:
    """Check if one of the cells collidates with another.

    Args:
        cells (list(Cell)): Cells which shall be checked

    Returns:
        bool: True if cells collidate.

    """
    for i in range(len(cells) - 1):
        if drc_collidates(cells[i], cells[i + 1 :]):
            return True

    return False


def drc_magic_all(name: str) -> int:
    """Check if the placement with name <name>, has DRC errors.

    Args:
        name (str): Name of the placement.

    Raises:
        FileNotFoundError: If the .mag file of the placement can't be found.

    Returns:
        int: Number of DRC errors

    """
    # check if Placement exists
    if os.path.exists(f"magic/placement/{name}.mag"):
        # if DRC folder exists delete it
        if os.path.exists("magic/placement/drc_result"):
            shutil.rmtree("magic/placement/drc_result")

        # make the DRC folder
        os.makedirs("magic/placement/drc_result")
        act_dir = os.getcwd()
        os.chdir("magic/placement/drc_result")

        # perform the DRC check in magic.
        os.system(f"bash {act_dir}/magic/magic_drc.sh ../{name}.mag")

        last_line = subprocess.check_output(["tail", "-1", f"{name}.magic.drc.rpt"])

        drc_errors = eval(last_line)

        os.chdir(act_dir)
        return drc_errors
    else:
        raise FileNotFoundError


def drc_magic_check_cell(layout_name: str, cell: Cell) -> int:
    """Check if the cell <cell> of placement with name <name>, has DRC errors.

    Args:
        layout_name (str): Name of the placement.
        cell (Cell): Cell to check DRC.

    Raises:
        FileNotFoundError: If the .mag file of the placement can't be found.

    Returns:
        int: Number of DRC errors

    """
    # check if Placement exists
    if os.path.exists(f"magic/placement/{layout_name}.mag"):
        # if DRC folder exists delete it
        if os.path.exists("magic/placement/drc_result"):
            shutil.rmtree("magic/placement/drc_result")

        box = cell.get_bounding_box()

        # make the DRC folder
        os.makedirs("magic/placement/drc_result")
        act_dir = os.getcwd()
        os.chdir("magic/placement/drc_result")
        os.system(
            f"bash {act_dir}/magic/magic_drc.sh -b {box[0]} -b {box[1]} -b {box[2]} -b {box[3]} ../{layout_name}.mag"
        )

        last_line = subprocess.check_output([
            "tail",
            "-1",
            f"{layout_name}.magic.drc.rpt",
        ])

        drc_errors = eval(last_line)

        os.chdir(act_dir)
        return drc_errors
    else:
        raise FileNotFoundError
