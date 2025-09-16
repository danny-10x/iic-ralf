# RALF - Reinforcement Learning assisted Automated analog Layout design Flow

**Please read and cite this paper about RALF: Jakob Ratschenberger, Harald Pretl. RALF: A Reinforcement Learning Assisted Automated Analog Layout Design Flow. TechRxiv. June 03, 2024.
DOI: 10.36227/techrxiv.171742468.82130474/v1 [link](https://www.techrxiv.org/users/684857/articles/1005136-ralf-a-reinforcement-learning-assisted-automated-analog-layout-design-flow).**


As part of a master's thesis, at the Institute for Integrated Circuits (IIC), Johannes Kepler University, Linz, Austria,
an automated analog layout design flow was developed.

<p align="center">
    <img src="images/RALF_methodology.png" width="600" />
<p/>

The input of the flow is a netlist in the SPICE format composed of devices using the SkyWater Technologies SKY130 process design kit. Optionally, a file in the json-format that contains information for the routing task, like minimum wire widths, can also be specified. From the netlist, the circuit is captured and converted into an internal data structure, that is capable for the tasks of the remaining flow. The subsequent stage annotates devices which are forming smaller circuits and match those in a precompiled database. Thus, these precompiled circuits allow the finding of differential pairs, differential loads, cross-coupled pairs, and in series connected resistors, so called R-strings. 
From the annotated circuit, the primitive cells are instantiated, by the use of the parametrized cell generator available in [MAGIC](https://github.com/RTimothyEdwards/magic). The positions of the cells in the layout are then found by using a reinforcement learning algorithm (or optionally by using a simulated annealing algorithm), such that they minimize a cost function based on the estimated total wire length and routing congestion. After the placement is fixed, a two stage routing-algorithm connects the devices. 
The first stage is a wire-planning algorithm which plans the routes on a rough tile-based grid and provides a guidance to the second stage, which is a detailed router. That one lays out the actual resources by using a gridless approach based on obstacle expansion. The output of the whole flow is a .mag-file which contains the placement, and a Tcl script for generating the routing in [MAGIC](https://github.com/RTimothyEdwards/magic).


### Net-rules file
A net-rules file contains information for the routing stage.
 - To specify the needed minimum width of a nets wires put the lines
 ```
 ["MinNetWireWidth",
    {
        "net" : <SubCircuit_Instance>.<Net_name>,
        "min_width" : <min_width>
    }
 ]
 ```
 If the net is located in the top-circuit the prefix `<SubCircuit_Instance>.` hasn't to be specified. The variable `<min_width>` defines the minimum width of the wire, whereby the unit of the width is in $\lambda = 10\mathrm{nm}$. 

 E.g. for the netlist
 ```
    x1 Vin1 Vout Vdd Vss buf
    XR1 Vin Vin1 Vss sky130_fd_pr__res_xhigh_po_0p35 L=2 mult=1 m=1
    XC1 Vin1 Vss sky130_fd_pr__cap_mim_m3_1 W=4 L=4 MF=1 m=1

    .subckt buf A Y Vdd Vss
    x1 A Y1 Vdd Vss inv
    x2 Y1 Y Vdd Vss inv
    .ends

    .subckt inv A Y Vdd Vss
    XM1 Y A Vss Vss sky130_fd_pr__nfet_01v8 L=1 W=1 nf=1 m=1
    XM2 Y A Vdd Vdd sky130_fd_pr__pfet_01v8 L=1 W=3 nf=3 m=1
    .ends
    .end
```
  the net `Vout1` gets accessed by 
 ```
 ["MinNetWireWidth",
    {
        "net" : Vout1,
        "min_width" : 20
    }
 ]
 ``` 

 The net `Y1` in the sub-circuit `buf` of device `x1` gets accessed by
 ```
 ["MinNetWireWidth",
    {
        "net" : x1.Y1,
        "min_width" : 20
    }
 ]
 ``` 

# Example - Differential amplifier 

In the following, the layout generation flow for the circuit `circuits/examples/DiffAmp.spice`, will be presented.

## Schematic

<p align="center">
    <img src="images/Example/DiffAmp_circuit.png" width="300" />
<p/>

## Placement
Run for example
```
python3 main_RP_placement.py
```
and place the circuit in Magic, per
```
python3 main_place_circuit.py
```

Resulting placement:
<p align="center">
    <img src="images/Example/DiffAmp_placement1.png" width="300" />
    <img src="images/Example/DiffAmp_placement2.png" width="300" />
<p/>    


## Routing
Run 
```
python3 main_routing.py
```
and show the routing in Magic per
```
python3 main_place_route_circuit.py
```
Resulting routing:
<p align="center">
    <img src="images/Example/DiffAmp_routing1.png" width="300" />
    <img src="images/Example/DiffAmp_routing2.png" width="300" />
<p/> 
