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

import numpy as np

from magic.cell import Cell


def cell_slide(cells: tuple[Cell, ...]):
    """Slide the cell, such that they not overlap, and all placement-rules are satisfied.

    Args:
        cells (List[Cell]): List of cells, to be slided.

    """
    # calculate bounding box of all cells
    bound = cells[0].get_bounding_box()

    for i in range(1, len(cells)):
        bound_i = cells[i].get_bounding_box()

        # get min L and B
        for k in range(2):
            bound[k] = min(bound[k], bound_i[k])

        # get max T and R
        for k in range(2, 4):
            bound[k] = max(bound[k], bound_i[k])

    # calculate mean-point of bounding box
    mean_point_bound = np.array(((bound[0] + bound[2]) / 2, (bound[1] + bound[3]) / 2))

    # get the reference cell (cell which is nearest to the mean-point of the bounding box)
    reference_cell = cells[0]

    for i in range(1, len(cells)):
        mean_point_cell_i = np.array(cells[i].center_point)
        mean_point_ref = np.array(reference_cell.center_point)

        if np.linalg.norm(mean_point_cell_i - mean_point_bound) < np.linalg.norm(
            mean_point_ref - mean_point_bound
        ):
            reference_cell = cells[i]

    unplaced_cells = copy.copy(cells)
    unplaced_cells.remove(reference_cell)
    placed_cells = [reference_cell]

    # sort unplaced cells according to the rel. dist. to the ref. cell
    unplaced_cells = sorted(
        unplaced_cells,
        key=lambda cell: np.linalg.norm(
            np.array(cell.center_point) - np.array(reference_cell.center_point)
        ),
    )

    while unplaced_cells:
        current_cell = unplaced_cells.pop(0)

        last_dir = -1
        for p_cell in placed_cells:
            # bound_p = p_cell.get_bounding_box()
            bound_p = p_cell.get_placement_bounding_box(current_cell)
            bound_c = current_cell.get_bounding_box()

            ex1 = bound_c[0] - bound_p[2]  # Rightward slide dist.
            ex2 = bound_c[2] - bound_p[0]  # Leftward slide dist.
            ex3 = bound_c[3] - bound_p[1]  # Downward slide dist.
            ex4 = bound_c[1] - bound_p[3]  # Upward slide dist.

            if ex1 >= 0 or ex2 <= 0 or ex3 <= 0 or ex4 >= 0:  # cells do not overlap
                # get optimal slide direction and distance
                ex = np.array([
                    (abs(ex1), 0),
                    (abs(ex2), 1),
                    (abs(ex3), 2),
                    (abs(ex4), 3),
                ])

                if last_dir >= 0 and last_dir <= 3:
                    # EX = np.delete(EX, int(last_DIR), 0)
                    min_ex = ex[last_dir, 0]
                    new_dir = ex[last_dir, 1]

                else:
                    ex = ex[ex[:, 0].argsort()]

                    min_ex = ex[0, 0]
                    new_dir = ex[0, 1]

                last_dir = int(new_dir)
                pass
            else:  # cells do overlap
                # get optimal slide direction and distance
                ex = np.array([
                    (abs(ex1), 0),
                    (abs(ex2), 1),
                    (abs(ex3), 2),
                    (abs(ex4), 3),
                ])

                if last_dir >= 0 and last_dir <= 3:
                    # EX = np.delete(EX, int(last_DIR), 0)
                    min_ex = ex[last_dir, 0]
                    new_dir = ex[last_dir, 1]

                else:
                    ex = ex[ex[:, 0].argsort()]

                    min_ex = ex[0, 0]
                    new_dir = ex[0, 1]

                last_dir = int(new_dir)

                if new_dir == 0:  # move rightward
                    dist = (min_ex, 0)
                elif new_dir == 1:  # move leftward
                    dist = (-min_ex, 0)
                elif new_dir == 2:  # move downward
                    dist = (0, -min_ex)
                else:  # move upward
                    dist = (0, min_ex)

                current_cell._move(dist)

        placed_cells.append(current_cell)
        # sort placed cells according to the rel. dist. to the ref. cell
        placed_cells = sorted(
            placed_cells,
            key=lambda cell: np.linalg.norm(
                np.array(cell.center_point) - np.array(reference_cell.center_point)
            ),
        )


def cell_slide2(cells: tuple[Cell, ...]):
    """Slide the cell, such that they not overlap, and all placement-rules are satisfied.

    Args:
        cells (List[Cell]): List of cells, to be slided.

    """
    # calculate bounding box of all cells
    bound = cells[0].get_bounding_box()

    for i in range(1, len(cells)):
        bound_i = cells[i].get_bounding_box()

        # get min L and B
        for k in range(2):
            bound[k] = min(bound[k], bound_i[k])

        # get max T and R
        for k in range(2, 4):
            bound[k] = max(bound[k], bound_i[k])

    # calculate mean-point of bounding box
    mean_point_bound = np.array(((bound[0] + bound[2]) / 2, (bound[1] + bound[3]) / 2))

    # get the reference cell (cell which is nearest to the mean-point of the bounding box)
    reference_cell = cells[0]

    for i in range(1, len(cells)):
        mean_point_cell_i = np.array(cells[i].center_point)
        mean_point_ref = np.array(reference_cell.center_point)

        if np.linalg.norm(mean_point_cell_i - mean_point_bound) < np.linalg.norm(
            mean_point_ref - mean_point_bound
        ):
            reference_cell = cells[i]

    unplaced_cells = copy.copy(cells)
    unplaced_cells.remove(reference_cell)
    placed_cells = [reference_cell]

    # sort unplaced cells according to the rel. dist. to the ref. cell
    unplaced_cells = sorted(
        unplaced_cells,
        key=lambda cell: np.linalg.norm(
            np.array(cell.center_point) - np.array(reference_cell.center_point)
        ),
    )

    while unplaced_cells:
        current_cell = unplaced_cells.pop(0)

        for p_cell in placed_cells:
            bound_p = p_cell.get_bounding_box()
            bound_c = current_cell.get_bounding_box()

            ex1 = bound_c[0] - bound_p[2]  # Rightward slide dist.
            ex2 = bound_c[2] - bound_p[0]  # Leftward slide dist.
            ex3 = bound_c[3] - bound_p[1]  # Downward slide dist.
            ex4 = bound_c[1] - bound_p[3]  # Upward slide dist.

            if ex1 >= 0 or ex2 <= 0 or ex3 <= 0 or ex4 >= 0:  # cells do not overlap
                pass
            else:  # cells do overlap
                # get optimal slide direction and distance
                ex = np.array([
                    (abs(ex1), 0),
                    (abs(ex2), 1),
                    (abs(ex3), 2),
                    (abs(ex4), 3),
                ])

                ref_coord = np.array(reference_cell.center_point)
                cur_coord = np.array(current_cell.center_point)

                v_0 = cur_coord - ref_coord
                v_r = np.array((ex[0, 0], 0))
                v_l = np.array((-ex[1, 0], 0))
                v_d = np.array((0, -ex[2, 0]))
                v_u = np.array((0, ex[3, 0]))

                del_dir = []
                if np.dot(v_0, v_r) < 0:
                    del_dir.append(0)

                if np.dot(v_0, v_l) < 0:
                    del_dir.append(1)

                if np.dot(v_0, v_d) < 0:
                    del_dir.append(2)

                if np.dot(v_0, v_u) < 0:
                    del_dir.append(3)

                ex = np.delete(ex, del_dir, axis=0)

                ex = ex[ex[:, 0].argsort()]
                min_ex = ex[0, 0]
                new_dir = ex[0, 1]

                if new_dir == 0:  # move rightward
                    dist = (min_ex, 0)
                elif new_dir == 1:  # move leftward
                    dist = (-min_ex, 0)
                elif new_dir == 2:  # move downward
                    dist = (0, -min_ex)
                else:  # move upward
                    dist = (0, min_ex)

                current_cell._move(dist)

        placed_cells.append(current_cell)
        # sort placed cells according to the rel. dist. to the ref. cell
        placed_cells = sorted(
            placed_cells,
            key=lambda cell: np.linalg.norm(
                np.array(cell.center_point) - np.array(reference_cell.center_point)
            ),
        )


def cell_slide3(cells: tuple[Cell, ...]):
    """Slide the cell, such that they not overlap, and all placement-rules are satisfied.

    Args:
        cells (List[Cell]): List of cells, to be slided.

    """
    # calculate the bounding box over all cells
    bound = cells[0].get_bounding_box()

    for i in range(1, len(cells)):
        bound_i = cells[i].get_bounding_box()

        # get min L and B
        for k in range(2):
            bound[k] = min(bound[k], bound_i[k])

        # get max T and R
        for k in range(2, 4):
            bound[k] = max(bound[k], bound_i[k])

    # calculate mean-point of the bounding box
    mean_point_bound = np.array(((bound[0] + bound[2]) / 2, (bound[1] + bound[3]) / 2))

    # get the reference cell (cell which is nearest to the mean-point of the bounding box)
    reference_cell = cells[0]

    # print(f"Reference cell: {reference_cell._name}")

    for i in range(1, len(cells)):
        mean_point_cell_i = np.array(cells[i].center_point)
        mean_point_ref = np.array(reference_cell.center_point)

        if np.linalg.norm(mean_point_cell_i - mean_point_bound) < np.linalg.norm(
            mean_point_ref - mean_point_bound
        ):
            reference_cell = cells[i]

    # store placed and unplaced cells
    unplaced_cells = copy.copy(cells)
    unplaced_cells.remove(reference_cell)
    # place the reference cell
    placed_cells = [reference_cell]

    # sort unplaced cells according to the rel. dist. to the ref. cell
    unplaced_cells = sorted(
        unplaced_cells,
        key=lambda cell: np.linalg.norm(
            np.array(cell.center_point) - np.array(reference_cell.center_point)
        ),
    )

    # while the are unplaced cells
    while unplaced_cells:
        # get the current cell
        # cell which is nearest to the placed cells
        current_cell = unplaced_cells.pop(0)
        # print(f"Sliding cell {current_cell._name}")

        i = 0
        # iterate over the placed cells
        while i < len(placed_cells):
            # get actual placed cell
            p_cell = placed_cells[i]
            # increase the bounding box of the current cell
            # to satisfy the placement rules between the current
            # cell and the actual placed cell
            bound_p = p_cell.get_placement_bounding_box(current_cell)
            bound_c = current_cell.get_bounding_box()

            # get the slide directions
            ex1 = bound_c[0] - bound_p[2]  # Rightward slide dist.
            ex2 = bound_c[2] - bound_p[0]  # Leftward slide dist.
            ex3 = bound_c[3] - bound_p[1]  # Downward slide dist.
            ex4 = bound_c[1] - bound_p[3]  # Upward slide dist.

            # check if the cells overlap
            if ex1 >= 0 or ex2 <= 0 or ex3 <= 0 or ex4 >= 0:  # cells do not overlap
                i += 1
            else:  # cells do overlap
                # set i to 0
                # -> at the next iteration, start to check
                # if the cell overlaps with one of the placed cells
                i = 0

                # get optimal slide direction and distance
                ex = np.array([
                    (abs(ex1), 0),
                    (abs(ex2), 1),
                    (abs(ex3), 2),
                    (abs(ex4), 3),
                ])

                ref_coord = np.array(reference_cell.center_point)
                cur_coord = np.array(current_cell.center_point)

                # vector from current cell to reference cell
                v_0 = cur_coord - ref_coord
                # vector for the rightward slide
                v_r = np.array((ex[0, 0], 0))
                # vector for the leftward slide
                v_l = np.array((-ex[1, 0], 0))
                # vector for the downward slide
                v_d = np.array((0, -ex[2, 0]))
                # vector for the upward slide
                v_u = np.array((0, ex[3, 0]))

                # get the non-feasible slide directions
                # a slide direction isn't feasible
                # if the slide isn't outwards
                del_dir = []
                if np.dot(v_0, v_r) < 0:
                    del_dir.append(0)

                if np.dot(v_0, v_l) < 0:
                    del_dir.append(1)

                if np.dot(v_0, v_d) < 0:
                    del_dir.append(2)

                if np.dot(v_0, v_u) < 0:
                    del_dir.append(3)

                # delete the non-feasible directions
                # print(f"Deleting dirs {del_dir}")
                ex = np.delete(ex, del_dir, axis=0)

                # get the optimal feasible slide direction
                # optimal -> slide which has the lowest "slide distance"
                ex = ex[ex[:, 0].argsort()]
                min_ex = ex[0, 0]
                new_dir = ex[0, 1]

                if new_dir == 0:  # move rightward
                    dist = (min_ex, 0)
                elif new_dir == 1:  # move leftward
                    dist = (-min_ex, 0)
                elif new_dir == 2:  # move downward
                    dist = (0, -min_ex)
                else:  # move upward
                    dist = (0, min_ex)

                current_cell._move(dist)

        placed_cells.append(current_cell)
        # sort placed cells according to the rel. dist. to the ref. cell
        placed_cells = sorted(
            placed_cells,
            key=lambda cell: np.linalg.norm(
                np.array(cell.center_point) - np.array(reference_cell.center_point)
            ),
        )
