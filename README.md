# Research Project CSE3000 2023 of TU Delft: Distributed CNOT gate experimental setup

This repository links to the work of students for the Research project course of the CSE bachelor at TU Delft.

The experimental setup was run on the SDK called [SquidASM](https://github.com/QuTech-Delft/squidasm) at 25-06-2023. Any changes to the repository or installation procedure overtime could result in the program not working properly. 

## Installation

The experimental setup runs on [SquidASM](https://github.com/QuTech-Delft/squidasm). Start by completing the installation procedure instructed on their [github page](https://github.com/QuTech-Delft/squidasm).

Run the following command in the squidasm direcory:

```
git clone https://github.com/woutvanloevezijn/Distributed-CNOT-gate-application-experimental-setup.git
```

## Usage

The `run_simulation.py` file contains and explains most of the configuration parameters mentioned in the paper. It is also the script which needs to be run in order to generate the desired plot and data file. 

```
python3 run_simulation.py
```

`application.py` stores the application of the control and target node which is used for to calculate the performance metrics. This application is based an example in the [NetQASM repository](https://github.com/QuTech-Delft/netqasm/tree/develop/netqasm/examples/apps/dist_cnot).

The `config.yaml`, `depolarise_link_config.yaml` and `heralded_link_config.yaml` contain the values for the perfect configuration of the stacks an link. However, they could be tweaked if necessary. For example for the parameter p_loss_length, the length parameter was set to 100. In `config.yaml`, `generic` needs to be replaced with `nv ` in `stacks:` and `depolarise` needs to be replaced with `heralded` in `links:` when changing the stack and link and vice versa.

`scraper.py` is a simple tool to calculate the sensitivity of all data files in a directory and print them to the terminal. The `direcory` value could be the string in which all data files are stored.

```
python3 scraper.py
```
