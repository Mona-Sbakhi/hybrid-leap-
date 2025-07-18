import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import argparse
import numpy as np
import copy

def set_random_seed(seed: Optional[int] = None):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def create_nodes(num_nodes: int, field_size: Tuple[int, int]) -> List[Dict]:
    """Create nodes with random positions and initial energy."""
    nodes = []
    for i in range(num_nodes):
        node = {
            'id': i,
            'pos': (random.uniform(0, field_size[0]), random.uniform(0, field_size[1])),
            'energy': round(1.0 + 0.8 * random.random(), 2),  # Between 1.0 and 1.8 J as per paper
            'is_CH': False,
            'assigned_to': None,
            'alive': True
        }
        nodes.append(node)
    return nodes

def get_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def select_cluster_heads(nodes: List[Dict], ch_probability: float, bs_location: Tuple[int, int], field_size: Tuple[int, int]) -> List[Dict]:
    """Select cluster heads based on probability, residual energy, and proximity to BS."""
    cluster_heads = []
    for node in nodes:
        if node['alive']:
            # Composite metric: higher energy and closer to BS increases probability
            dist_to_bs = get_distance(node['pos'], bs_location)
            prob_adjust = (node['energy'] / 1.8) * (1 / (1 + dist_to_bs / max(field_size)))
            if random.random() < ch_probability * prob_adjust:
                node['is_CH'] = True
                cluster_heads.append(node)
    return cluster_heads

def assign_nodes_to_ch(nodes: List[Dict], cluster_heads: List[Dict]):
    """Assign each non-CH node to the nearest cluster head."""
    for node in nodes:
        if not node['is_CH'] and node['alive'] and cluster_heads:
            closest_ch = min(cluster_heads, key=lambda ch: get_distance(node['pos'], ch['pos']))
            node['assigned_to'] = closest_ch['id']

def create_pegasis_chain(members: List[Dict]) -> List[Dict]:
    """Form a PEGASIS chain among the given members, considering energy."""
    if not members:
        return []
    chain = [members[0]]
    visited = {members[0]['id']}
    while len(chain) < len(members):
        last = chain[-1]
        next_node = min(
            [n for n in members if n['id'] not in visited and n['alive']],
            key=lambda n: get_distance(last['pos'], n['pos']) / n['energy'],
            default=None
        )
        if next_node:
            chain.append(next_node)
            visited.add(next_node['id'])
        else:
            break
    return chain

def build_pegasis_chains(nodes: List[Dict], cluster_heads: List[Dict]) -> Dict[int, List[int]]:
    """Build PEGASIS chains for each cluster head group."""
    pegasis_chains = {}
    for ch in cluster_heads:
        members = [n for n in nodes if n['assigned_to'] == ch['id'] and n['alive']]
        full_group = [ch] + members
        chain = create_pegasis_chain(full_group)
        pegasis_chains[ch['id']] = [n['id'] for n in chain]
    return pegasis_chains

# TDMA Simulation: Simple model to assign time slots and simulate transmission
def simulate_tdma_transmission(chain: List[int], nodes: List[Dict], bs_location: Tuple[int, int]):
    """Simulate TDMA scheduling and data transmission along the chain, updating energy."""
    if not chain:
        return
    # Assign time slots: one slot per node in chain
    num_slots = len(chain)
    # Simulate transmission: each node sends to next, last to BS
    for i in range(len(chain) - 1):
        sender_id = chain[i]
        receiver_id = chain[i+1]
        sender = next(n for n in nodes if n['id'] == sender_id)
        receiver = next(n for n in nodes if n['id'] == receiver_id)
        dist = get_distance(sender['pos'], receiver['pos'])
        energy_cost = 0.05 * dist  # As per paper: proportional to d (though typically d^2)
        sender['energy'] -= energy_cost
        if sender['energy'] <= 0:
            sender['energy'] = 0
            sender['alive'] = False
    
    # Last node in chain (CH) sends to BS
    last_id = chain[-1]
    last_node = next(n for n in nodes if n['id'] == last_id)
    dist_to_bs = get_distance(last_node['pos'], bs_location)
    energy_cost_to_bs = 0.05 * dist_to_bs
    last_node['energy'] -= energy_cost_to_bs
    if last_node['energy'] <= 0:
        last_node['energy'] = 0
        last_node['alive'] = False

def print_simulation_results(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], round_num: int):
    alive_nodes = [n for n in nodes if n['alive']]
    print(f"Round {round_num}:")
    print("Total Alive Nodes:", len(alive_nodes))
    print("Cluster Heads:", [ch['id'] for ch in cluster_heads])
    print("Average Energy:", round(sum(n['energy'] for n in alive_nodes) / len(alive_nodes) if alive_nodes else 0, 2))
    print("\nPEGASIS Chains:")
    for ch_id, chain in pegasis_chains.items():
        print(f"CH {ch_id}: {chain}")
        print(f"Chain Length: {len(chain)}")
        chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain]
        avg_chain_energy = round(sum(n['energy'] for n in chain_nodes) / len(chain_nodes), 2)
        print(f"Average Energy in Chain: {avg_chain_energy}")

def plot_network(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], field_size: Tuple[int, int], bs_location: Tuple[int, int], save_plot: str, round_num: int):
    plt.figure(figsize=(10, 10))
    plt.title(f"Hybrid-LEAP Network (Round {round_num})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # Plot alive normal nodes
    for node in nodes:
        if node['alive'] and not node['is_CH']:
            plt.plot(node['pos'][0], node['pos'][1], 'ko', markersize=7)
            plt.text(node['pos'][0] + 1, node['pos'][1] + 1, str(node['id']), fontsize=8)
    # Plot dead nodes
    for node in nodes:
        if not node['alive']:
            plt.plot(node['pos'][0], node['pos'][1], 'kx', markersize=7)
    # Plot cluster heads
    for ch in cluster_heads:
        if ch['alive']:
            plt.plot(ch['pos'][0], ch['pos'][1], 'r*', markersize=15)
            plt.text(ch['pos'][0] + 1, ch['pos'][1] + 1, str(ch['id']), fontsize=9, color='red')
    # Plot base station
    plt.plot(bs_location[0], bs_location[1], 'bs', markersize=14)
    # Plot PEGASIS chains
    colors = ['g', 'b', 'm', 'c', 'y']
    for idx, (ch_id, chain) in enumerate(pegasis_chains.items()):
        chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain if next((n for n in nodes if n['id'] == nid), None)['alive']]
        color = colors[idx % len(colors)]
        for i in range(len(chain_nodes) - 1):
            x_vals = [chain_nodes[i]['pos'][0], chain_nodes[i+1]['pos'][0]]
            y_vals = [chain_nodes[i]['pos'][1], chain_nodes[i+1]['pos'][1]]
            plt.plot(x_vals, y_vals, color + '-', linewidth=2)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Alive Node'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='k', markersize=8, label='Dead Node'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=15, label='Cluster Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='b', markersize=12, label='Base Station'),
        Line2D([0], [0], color='g', lw=2, label='PEGASIS Link')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.xlim(0, field_size[0] + 10)
    plt.ylim(0, field_size[1] + 10)
    plt.grid(True)
    plt.savefig(f"{save_plot}_round_{round_num}.png")
    plt.close()

# Function to reset CH status for new round
def reset_ch_status(nodes: List[Dict]):
    for node in nodes:
        node['is_CH'] = False
        node['assigned_to'] = None

def run_simulation(
    num_nodes: int = 20,
    field_size: Tuple[int, int] = (100, 100),
    ch_probability: float = 0.2,
    bs_location: Tuple[int, int] = (100, 100),
    seed: Optional[int] = None,
    save_plot: str = 'plot',
    num_rounds: int = 20
):
    """Run the Hybrid-LEAP simulation over multiple rounds."""
    set_random_seed(seed)
    nodes = create_nodes(num_nodes, field_size)
    metrics = {'residual_energy': [], 'alive_nodes': [], 'avg_chain_energy': [], 'chain_lengths': []}
    
    for round_num in range(1, num_rounds + 1):
        reset_ch_status(nodes)
        cluster_heads = select_cluster_heads(nodes, ch_probability, bs_location, field_size)
        assign_nodes_to_ch(nodes, cluster_heads)
        pegasis_chains = build_pegasis_chains(nodes, cluster_heads)
        
        # Simulate TDMA and transmission for each chain
        for ch_id, chain in pegasis_chains.items():
            simulate_tdma_transmission(chain, nodes, bs_location)
        
        # Collect metrics
        alive_nodes = sum(1 for n in nodes if n['alive'])
        total_energy = sum(n['energy'] for n in nodes if n['alive'])
        chain_energies = []
        chain_lens = []
        for chain in pegasis_chains.values():
            chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain if next((n for n in nodes if n['id'] == nid), None)['alive']]
            if chain_nodes:
                chain_energies.append(sum(n['energy'] for n in chain_nodes) / len(chain_nodes))
                chain_lens.append(len(chain_nodes))
        
        avg_chain_energy = np.mean(chain_energies) if chain_energies else 0
        avg_chain_length = np.mean(chain_lens) if chain_lens else 0
        
        metrics['residual_energy'].append(total_energy)
        metrics['alive_nodes'].append(alive_nodes)
        metrics['avg_chain_energy'].append(avg_chain_energy)
        metrics['chain_lengths'].append(avg_chain_length)
        
        print_simulation_results(nodes, cluster_heads, pegasis_chains, round_num)
        plot_network(nodes, cluster_heads, pegasis_chains, field_size, bs_location, save_plot, round_num)
        
        if alive_nodes == 0:
            print("All nodes dead. Ending simulation.")
            break
    
    # Plot overall metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['residual_energy'])
    plt.title('Total Residual Energy per Round')
    plt.subplot(2, 2, 2)
    plt.plot(metrics['alive_nodes'])
    plt.title('Alive Nodes per Round')
    plt.subplot(2, 2, 3)
    plt.plot(metrics['avg_chain_energy'])
    plt.title('Average Chain Energy per Round')
    plt.subplot(2, 2, 4)
    plt.plot(metrics['chain_lengths'])
    plt.title('Average Chain Length per Round')
    plt.savefig(f"{save_plot}_overall_metrics.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid-LEAP simulation with command-line parameters.")
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of nodes")
    parser.add_argument('--field_size', type=int, nargs=2, default=[100, 100], help="Field size (width height)")
    parser.add_argument('--ch_probability', type=float, default=0.2, help="Cluster head probability")
    parser.add_argument('--bs_location', type=int, nargs=2, default=[100, 100], help="Base station location (x y)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed (optional)")
    parser.add_argument('--save_plot', type=str, default='plot', help="Filename prefix to save the plots")
    parser.add_argument('--num_rounds', type=int, default=20, help="Number of simulation rounds")
    args = parser.parse_args()

    # Convert lists to tuples
    field_size_tuple = tuple(args.field_size)
    bs_location_tuple = tuple(args.bs_location)

    run_simulation(
        num_nodes=args.num_nodes,
        field_size=field_size_tuple,
        ch_probability=args.ch_probability,
        bs_location=bs_location_tuple,
        seed=args.seed,
        save_plot=args.save_plot,
        num_rounds=args.num_rounds
    )