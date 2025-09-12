# ========================================================================
#
#   Script to run automated analog layout flow.
#
# SPDX-FileCopyrightText: 2025 Danny Boby
# TenX Semiconductors, Inc.
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
import argparse
import os
import sys


def main():
    """Run the automated analog layout flow."""
    # Check if the operating system is Unix-like (posix)
    if os.name != "posix":
        print(
            "Error: This script is designed to run on a Unix-like system (e.g., Linux or macOS)."
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run automated analog layout flow.")
    parser.add_argument(
        "--circuit_file",
        type=str,
        default="circuits/examples/DiffAmp.spice",
        help="Input spice-netlist",
    )
    parser.add_argument(
        "--circuit_name", type=str, default="DiffAmp", help="Name of the top-circuit"
    )

    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
