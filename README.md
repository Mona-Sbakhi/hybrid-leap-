# Hybrid-LEAP Simulation

This repository contains a Python script (`hybrid_leap_simulation.py`) for simulating a Hybrid-LEAP (Low-Energy Adaptive Clustering Hierarchy with PEGASIS) protocol in wireless sensor networks. The simulation creates nodes in a field, selects cluster heads based on probability, assigns nodes to clusters, builds PEGASIS chains for energy-efficient data transmission, and visualizes the network topology.

The script outputs simulation results to the console (e.g., number of nodes, cluster heads, average energy, and PEGASIS chains) and saves a plot of the network as a PNG file.

## Prerequisites

- **Python**: Version 3.6 or higher (tested on Python 3.12).
- **Required Libraries**:
  - `matplotlib`: For plotting the network visualization. Install via pip:
    ```
    pip install matplotlib
    ```
  - Other dependencies (e.g., `random`, `math`, `argparse`) are part of the Python standard library.

No additional packages are needed beyond matplotlib.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mona-Sbakhi/hybrid-leap-.git
   cd hybrid-leap-
   ```

2. Install the required library:
    ```
   pip install matplotlib
   ```

## Usage

Run the simulation using the command-line interface. The script accepts various parameters to customize the simulation.

### Example Command
```
python hybrid_leap_simulation.py --num_nodes 30 --field_size 150 150 --ch_probability 0.3 --bs_location 120 120 --seed 42 --save_plot output1.png
```

This will:
- Generate simulation results in the console.
- Save a plot of the network to `output1.png`.

### Parameters
- **`--num_nodes`**: Number of nodes in the network (default: 20).
- **`--field_size`**: Field size as two integers (width height) (default: 100 100).
- **`--ch_probability`**: Probability for a node to become a cluster head (default: 0.2).
- **`--bs_location`**: Base station location as two integers (x y) (default: 100 100).
- **`--seed`**: Random seed for reproducibility (optional, default: None).
- **`--save_plot`**: Path to save the plot (default: plot.png).

## Benchmark Examples

Use these pre-defined parameter sets to benchmark the simulation. Copy and run them in your terminal.

```
python hybrid_leap_simulation.py --num_nodes 20 --field_size 100 100 --ch_probability 0.2 --bs_location 100 100 --seed 1 --save_plot benchmark1.png
```
```
python hybrid_leap_simulation.py --num_nodes 50 --field_size 200 200 --ch_probability 0.1 --bs_location 180 180 --seed 2 --save_plot benchmark2.png
```
```
python hybrid_leap_simulation.py --num_nodes 100 --field_size 300 300 --ch_probability 0.15 --bs_location 250 250 --seed 3 --save_plot benchmark3.png
```
```
python hybrid_leap_simulation.py --num_nodes 30 --field_size 120 120 --ch_probability 0.25 --bs_location 60 60 --seed 4 --save_plot benchmark4.png
```
```
python hybrid_leap_simulation.py --num_nodes 40 --field_size 150 100 --ch_probability 0.18 --bs_location 75 50 --seed 5 --save_plot benchmark5.png
```