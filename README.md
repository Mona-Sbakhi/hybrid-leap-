# Hybrid-LEAP / LEACH / PEGASIS Simulation

This repository contains a Python script (`hybrid_leap_simulation.py`) for simulating three key WSN routing protocols:
- **Hybrid-LEAP**: Combines LEACH clustering and intra-cluster PEGASIS chaining with TDMA scheduling.
- **LEACH**: Classic clustering protocol with random CH rotation.
- **PEGASIS**: Chain-based protocol with single chain and linear aggregation.

The simulation evaluates energy efficiency, packet delivery, latency, and node survivability over multiple rounds.

## Features

- Cluster head (CH) selection based on energy and proximity (Hybrid-LEAP / LEACH).
- PEGASIS chaining with energy-aware greedy formation.
- TDMA-based transmission simulation with packet loss probability.
- Support for adjustable packet size, node energy range, and round limits.
- Visualizes node status and network links per round.
- Outputs performance metrics and a `.csv` results file.

## Prerequisites

- **Python** 3.6+
- Required libraries:
  ```bash
  pip install matplotlib numpy pandas
  ```

## Installation

```bash
git clone https://github.com/Mona-Sbakhi/hybrid-leap-.git
cd hybrid-leap-
```

## Usage

Run the simulation:

```bash
python hybrid_leap_simulation.py --num_nodes 30 --field_size 150 150 --ch_probability 0.3 --bs_location 120 120 --seed 42 --save_plot output1 --num_rounds 25 --packet_size 4000 --packet_loss_prob 0.1
```

### Parameters

| Parameter             | Type     | Default   | Description |
|----------------------|----------|-----------|-------------|
| `--num_nodes`        | `int`    | `75`      | Number of sensor nodes |
| `--field_size`       | `int int`| `400 400` | Field dimensions in meters |
| `--ch_probability`   | `float`  | `0.1`     | CH selection base probability |
| `--bs_location`      | `int int`| `200 200` | Coordinates of the base station |
| `--seed`             | `int`    | `None`    | Random seed for reproducibility |
| `--save_plot`        | `str`    | `'plot'`  | Prefix for plot and result filenames |
| `--num_rounds`       | `int`    | `100`     | Number of simulation rounds |
| `--packet_loss_prob` | `float`  | `0.1`     | Per-hop packet loss probability |
| `--min_energy`       | `float`  | `0.1`     | Minimum node initial energy (Joules) |
| `--max_energy`       | `float`  | `1.0`     | Maximum node initial energy (Joules) |
| `--packet_size`      | `int`    | `4000`    | Packet size in bits |

Run `python hybrid_leap_simulation.py --help` for full options.

## Output

- **Console Logs**:
  - Alive node count
  - CH IDs
  - Average energy
  - Latency (s)
  - Packet delivery ratio
  - PEGASIS chains per CH (Hybrid-LEAP)

- **Saved Files**:
  - `plot_<protocol>_round_<N>.png`: Network topology per round
  - `plot_all_metrics_overall.png`: Plots for residual energy, latency, delivery, alive nodes
  - `plot_results.csv`: Round-by-round performance metrics for all protocols

## Benchmark Examples

```bash
python hybrid_leap_simulation.py --num_nodes 20 --field_size 100 100 --ch_probability 0.2 --bs_location 100 100 --seed 1 --save_plot bench1 --num_rounds 20
python hybrid_leap_simulation.py --num_nodes 50 --field_size 200 200 --ch_probability 0.1 --bs_location 180 180 --seed 2 --save_plot bench2 --num_rounds 20
python hybrid_leap_simulation.py --num_nodes 100 --field_size 300 300 --ch_probability 0.15 --bs_location 250 250 --seed 3 --save_plot bench3 --num_rounds 20
```

## Visualization Legend

- **Black Circle (`o`)**: Alive node  
- **Black Cross (`x`)**: Dead node  
- **Red Star (`*`)**: Cluster Head  
- **Blue Square (`s`)**: Base Station  
- **Green Line**: PEGASIS link

---

For questions or contributions, feel free to open an issue or PR.
