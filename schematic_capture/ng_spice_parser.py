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


class Parser:
    """Class to parse a ngspice netlist.

    The following transformations will be applied:
        - broken lines will be merged
        - empty lines will be deleted
        - white spaces at the start and end of each line will be deleted
        - comments will be deleted
    """

    def __init__(self, src: str) -> None:
        """Set up a ngspice netlist parser.

        Args:
            src (str): Netlist file.

        Raises:
            ValueError: If the src can't be read.

        """
        self._src = src

        try:
            with open(str(src)) as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError from e

        self.spc_netlist = self._merge_lines(lines)
        self.spc_netlist = self._delete_empty_lines(self.spc_netlist)
        self.spc_netlist = self._delete_start_end_whitespace(self.spc_netlist)
        self.spc_netlist = self._delete_comments(self.spc_netlist)

    def _merge_lines(self, lines):
        """Merge broken lines.

        Parameters
        ----------
        lines : list (str)
            Raw netlist.

        Returns
        -------
        merged_lines : list (str)
            Raw netlist with merged lines.

        """
        merged_lines = []
        for line_ in lines:
            if line_.startswith("+ "):
                last = merged_lines.pop(-1)
                last = last + line_[1:-1]
                merged_lines.append(last)
            else:
                merged_lines.append(line_)
        return merged_lines

    def _delete_start_end_whitespace(self, net: list[str]):
        """Remove whitespaces at the beginning and at the end.

        Parameters
        ----------
        net : list (str)
            Netlist.

        Returns
        -------
        new_net : list (str)
            Netlist.

        """
        new_net = []
        for line_ in net:
            new_net.append(line_.strip())
        return new_net

    def _delete_empty_lines(self, net: list[str]):
        """Remove empty lines from the netlist.

        Parameters
        ----------
        net : list (str)
            Netlist.

        Returns
        -------
        new_net : list (str)
            Netlist without empty lines.

        """
        new_net = []
        for line_ in net:
            if line_.strip():
                new_net.append(line_)

        return new_net

    def _delete_comments(self, net: list[str]):
        """Delete all comments in a netlist.

        Line comments: *
        End-of-Line comments: $ , ; , //

        Parameters
        ----------
        net : list (str)
            Netlist.

        Returns
        -------
        new_net : list (str)
            Netlist without comments.

        """
        new_net = []

        end_of_line_comment = ["$ ", "; ", "//"]

        for line_ in net:
            if line_.startswith("*"):  # comment line
                continue

            indx = -1
            for c in end_of_line_comment:
                indx = line_.find(c)
                if not (indx == -1):
                    break

            if not (indx == -1):
                new_net.append(line_[0:indx].rstrip())
                continue

            new_net.append(line_)

        return new_net

    def get_netlist(self) -> list[str]:
        """Get the parsed netlist.

        Returns:
            list[str]: Parsed netlist.

        """
        return self.spc_netlist
