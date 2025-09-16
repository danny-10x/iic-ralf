# Getting started
## Step 0: Prerequisites (Recommended)
- Use the [IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS) all in one docker container.

## Step 0.1: Prerequisites (Optional)
- [SKY130 PDK](https://github.com/google/skywater-pdk)
    - For easy installation checkout [volare](https://github.com/efabless/volare)
- [MAGIC](https://github.com/RTimothyEdwards/magic)
- Python >= 3.9 with the installed [requirements](https://github.com/JakobRat/RALF/edit/main/requirements.txt)
- Path to the sky130A pdk set under `$PDKPATH`, this can look like as follows
```
export PDKPATH=/home/pdks/sky130A
```

## Step 1: Clone the repository
```
git clone https://github.com/danny-10x/iic-ralf
```

## Step 2: Install uv and create a virtual environment
```
pip install uv
uv sync
source .venv/bin/activate
```

## Step 3: Add your circuits netlist
To design your circuit, add the circuits-netlist (only `.spice` formats are supported) to the `circuits` folder. 

### Netlist prerequisites
- The top-circuit isn't a subcircuit.
- The netlist only contains the devices
    - sky130_fd_pr__nfet_01v8
    - sky130_fd_pr__pfet_01v8
    - sky130_fd_pr__cap_mim_m3_1
    - sky130_fd_pr__cap_mim_m3_2
    - sky130_fd_pr__res_xhigh_po_0p35
- E.g. a valid netlist looks like
```
    x1 Vin Vout1 Vdd Vss inv
    x2 Vin2 Vout Vdd Vss inv
    XR1 Vout1 Vin2 Vss sky130_fd_pr__res_xhigh_po_0p35 L=2 mult=1 m=1
    XC1 Vin2 Vss sky130_fd_pr__cap_mim_m3_1 W=4 L=4 MF=1 m=1

    .subckt inv A Y Vdd Vss
    XM1 Y A Vss Vss sky130_fd_pr__nfet_01v8 L=1 W=1 nf=1 m=1
    XM2 Y A Vdd Vdd sky130_fd_pr__pfet_01v8 L=1 W=3 nf=3 m=1
    .ends
    .end
```

## Step 4: Do a placement
There are two supported placement mechanisms:
- Reinforcement learning based (`rl_placement.py`)
- Simulated annealing based (`rp_placement.py`)

To do a placement, adapt the global variables according to your circuit, and run the script in a shell.
The most valuable ones are
- `CIRCUIT_FILE`: Defines the input SPICE-netlist.
- `CIRCUIT_NAME`: Defines the name of the top-circuit and top-cell.
- `NET_RULES_FILE`: Defines the net-rules file in the json-format, to specify different net-widths. If not available set the variable to `None`.
- `N_PLACEMENTS`: Defines the total number of performed trial placements.

For the reinforcement learning based placement run:
```
python rl_placement.py
```
For the simulated annealing based placement run:
```
python rp_placement.py
```
The placed circuit will be stored under `placement_circuits/<circuit_name>_placement.pkl`.

## Step 5: View the placement in Magic
To view the placement in Magic run the script `place_circuit.py`.
```
python place_circuit.py
```
Don't forget to adapt the variable `CIRCUIT_NAME` to your circuits name!\
The generated Magic file of the placement will be located under `magic/placement/<CIRCUIT_NAME>.mag`.