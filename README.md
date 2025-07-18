# Hybrid-LEAP Simulation

This repository contains a Python script (`hybrid_leap_simulation.py`) for simulating a Hybrid-LEAP (Low-Energy Adaptive Clustering Hierarchy with PEGASIS) protocol in wireless sensor networks. The simulation creates nodes in a field, selects cluster heads based on a probability adjusted by residual energy and proximity to the base station, assigns nodes to clusters, builds PEGASIS chains for energy-efficient data transmission, simulates TDMA-scheduled transmissions with energy depletion, and visualizes the network topology over multiple rounds.

The script runs the simulation for a specified number of rounds, updating node energies after each round's transmissions and marking nodes as dead when energy reaches zero. It outputs detailed results to the console for each round (e.g., alive nodes, cluster heads, average energy, PEGASIS chains, chain lengths, and average chain energy) and saves plots of the network state per round, as well as an overall metrics summary plot.

## Prerequisites

- **Python**: Version 3.6 or higher (tested on Python 3.12).
- **Required Libraries**:
  - `matplotlib`: For plotting the network visualization and metrics.
  - `numpy`: For numerical computations and averaging metrics.
  - Install via pip:
    ```
    pip install matplotlib numpy
    ```
  - Other dependencies (e.g., `random`, `math`, `argparse`, `copy`) are part of the Python standard library.

No additional packages are needed beyond matplotlib and numpy.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mona-Sbakhi/hybrid-leap-.git
   cd hybrid-leap-
   ```

2. Install the required libraries:
   ```
   pip install matplotlib numpy
   ```

## Usage

Run the simulation using the command-line interface. The script accepts various parameters to customize the simulation, including the number of rounds and network configuration.

### Example Command
```
python hybrid_leap_simulation.py --num_nodes 30 --field_size 150 150 --ch_probability 0.3 --bs_location 120 120 --seed 42 --save_plot output1 --num_rounds 25
```

This will:
- Run the simulation for 25 rounds, generating console output and network plots for each round.
- Save network plots as `output1_round_<round_num>.png`.
- Save an overall metrics plot as `output1_overall_metrics.png`.

### Parameters
The script supports the following command-line parameters to configure the simulation:

- **`--num_nodes`** (int, default: 20):
  - Specifies the number of sensor nodes in the network. Nodes are randomly placed in the field and assigned initial energies between 1.0 and 1.8 Joules.

- **`--field_size`** (two integers: width height, default: 100 100):
  - Defines the dimensions of the simulation field (in meters) where nodes are randomly placed. Example: `--field_size 150 150` sets a 150x150 m² field.

- **`--ch_probability`** (float, default: 0.2):
  - Sets the base probability for a node to become a cluster head (CH). This probability is adjusted by each node’s residual energy and proximity to the base station, ensuring energy-efficient and BS-proximate nodes are prioritized.

- **`--bs_location`** (two integers: x y, default: 100 100):
  - Specifies the coordinates of the base station (BS) in the field, used for CH selection and data transmission. Example: `--bs_location 120 120` places the BS at (120, 120).

- **`--seed`** (int, optional, default: None):
  - Sets a random seed for reproducibility. If provided, ensures consistent node placement, energy assignments, and CH selections across runs. If set to `None`, results are non-deterministic.

- **`--save_plot`** (str, default: 'plot'):
  - Defines the filename prefix for saved plots. Network plots are saved as `<save_plot>_round_<round_num>.png` for each round, and the metrics summary is saved as `<save_plot>_overall_metrics.png`.

- **`--num_rounds`** (int, default: 20):
  - Specifies the number of simulation rounds. Each round involves CH selection, cluster assignment, chain formation, TDMA-scheduled transmission, and energy updates. The simulation stops early if all nodes deplete their energy.

To view all parameters and their defaults, run:
```
python hybrid_leap_simulation.py --help
```

## Benchmark Examples

Use these pre-defined parameter sets to benchmark the simulation over multiple rounds. Copy and run them in your terminal.

```
python hybrid_leap_simulation.py --num_nodes 20 --field_size 100 100 --ch_probability 0.2 --bs_location 100 100 --seed 1 --save_plot benchmark1 --num_rounds 20
```
```
python hybrid_leap_simulation.py --num_nodes 50 --field_size 200 200 --ch_probability 0.1 --bs_location 180 180 --seed 2 --save_plot benchmark2 --num_rounds 20
```
```
python hybrid_leap_simulation.py --num_nodes 100 --field_size 300 300 --ch_probability 0.15 --bs_location 250 250 --seed 3 --save_plot benchmark3 --num_rounds 20
```
```
python hybrid_leap_simulation.py --num_nodes 30 --field_size 120 120 --ch_probability 0.25 --bs_location 60 60 --seed 4 --save_plot benchmark4 --num_rounds 20
```
```
python hybrid_leap_simulation.py --num_nodes 40 --field_size 150 100 --ch_probability 0.18 --bs_location 75 50 --seed 5 --save_plot benchmark5 --num_rounds 20
```

## Output

- **Console Output**: For each round, includes:
  - Total number of alive nodes.
  - List of cluster heads.
  - Average energy of alive nodes.
  - PEGASIS chains, their lengths, and average energy per chain.
- **Network Plots (per round)**: Visual representations saved as `<save_plot>_round_<round_num>.png`, showing:
  - Alive normal nodes (black circles).
  - Dead nodes (black crosses).
  - Cluster heads (red stars).
  - Base station (blue square).
  - PEGASIS links (colored lines).
- **Overall Metrics Plot**: A summary figure saved as `<save_plot>_overall_metrics.png`, with subplots for:
  - Total residual energy per round.
  - Number of alive nodes per round.
  - Average chain energy per round.
  - Average chain length per round.

The simulation ends early if all nodes die before completing the specified rounds.

For questions or contributions, feel free to open an issue or pull request!