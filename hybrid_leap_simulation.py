import random
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy
import pandas as pd
from typing import List, Dict, Tuple, Optional

def set_random_seed(seed: Optional[int] = None):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def create_nodes(num_nodes: int, field_size: Tuple[int, int], min_energy: float, max_energy: float) -> List[Dict]:
    """Create nodes with random positions and initial energy."""
    nodes = []
    for i in range(num_nodes):
        node = {
            'id': i,
            'pos': (random.uniform(0, field_size[0]), random.uniform(0, field_size[1])),
            'energy': round(min_energy + (max_energy - min_energy) * random.random(), 2),
            'is_CH': False,
            'assigned_to': None,
            'alive': True
        }
        nodes.append(node)
    return nodes

def get_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def select_cluster_heads(nodes: List[Dict], ch_probability: float, bs_location: Tuple[int, int], field_size: Tuple[int, int], protocol: str, max_energy: float) -> List[Dict]:
    """Select cluster heads based on protocol (Hybrid-LEAP or LEACH)."""
    cluster_heads = []
    energy_threshold = 0.1 * max_energy
    if protocol == "hybrid-leap":
        attempts = 0
        while not cluster_heads and attempts < 20:
            cluster_heads = []
            for node in nodes:
                if node['alive'] and node['energy'] > energy_threshold:
                    dist_to_bs = get_distance(node['pos'], bs_location)
                    prob_adjust = (node['energy'] / max_energy) * (1 / (1 + dist_to_bs / max(field_size)))
                    if random.random() < ch_probability * prob_adjust:
                        node['is_CH'] = True
                        cluster_heads.append(node)
            attempts += 1
    elif protocol == "leach":
        attempts = 0
        while not cluster_heads and attempts < 20:
            cluster_heads = []
            for node in nodes:
                if node['alive'] and node['energy'] > energy_threshold and random.random() < ch_probability:
                    node['is_CH'] = True
                    cluster_heads.append(node)
            attempts += 1
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
    # Select starting node with highest energy/distance ratio
    start_node = max(members, key=lambda n: n['energy'] / (sum(get_distance(n['pos'], m['pos']) for m in members if m != n) or 0.001))
    chain = [start_node]
    visited = {start_node['id']}
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

def create_single_pegasis_chain(nodes: List[Dict]) -> List[int]:
    """Form a single PEGASIS chain for all alive nodes."""
    alive_nodes = [n for n in nodes if n['alive']]
    if not alive_nodes:
        return []
    return [n['id'] for n in create_pegasis_chain(alive_nodes)]

sensing_energy = 5e-9  # Energy for sensing per round per node (Joules)

def simulate_transmission(nodes: List[Dict], protocol: str, cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], bs_location: Tuple[int, int], packet_loss_prob: float, packet_size: int, e_elec: float = 50e-9, e_amp: float = 100e-12) -> Tuple[float, float]:
    """Simulate transmission for a protocol, updating energy and tracking latency/packet delivery."""
    total_latency = 0.0
    packets_sent = 0
    packets_delivered = 0

    # Apply sensing energy to all alive nodes
    for node in nodes:
        if node['alive']:
            node['energy'] = max(0, node['energy'] - sensing_energy * packet_size)
            if node['energy'] <= 0:
                node['alive'] = False

    if protocol == "hybrid-leap":
        max_chain_length = 0
        for ch_id, chain in pegasis_chains.items():
            if not chain:
                continue
            num_slots = len(chain)
            max_chain_length = max(max_chain_length, num_slots)
            packets_sent += num_slots
            for i in range(len(chain) - 1):
                sender_id = chain[i]
                receiver_id = chain[i + 1]
                sender = next(n for n in nodes if n['id'] == sender_id)
                receiver = next(n for n in nodes if n['id'] == receiver_id)
                dist = get_distance(sender['pos'], receiver['pos'])
                energy_cost_tx = (e_elec + e_amp * dist * dist) * packet_size
                energy_cost_rx = e_elec * packet_size
                sender['energy'] = max(0, sender['energy'] - energy_cost_tx)
                receiver['energy'] = max(0, receiver['energy'] - energy_cost_rx)
                if sender['energy'] <= 0:
                    sender['alive'] = False
                if receiver['energy'] <= 0:
                    receiver['alive'] = False
                if random.random() > packet_loss_prob:
                    packets_delivered += 1
            last_id = chain[-1]
            last_node = next(n for n in nodes if n['id'] == last_id)
            dist_to_bs = get_distance(last_node['pos'], bs_location)
            energy_cost_to_bs = (e_elec + e_amp * dist_to_bs * dist_to_bs) * packet_size
            last_node['energy'] = max(0, last_node['energy'] - energy_cost_to_bs)
            if last_node['energy'] <= 0:
                last_node['alive'] = False
            if random.random() > packet_loss_prob:
                packets_delivered += 1
        total_latency = np.mean([len(chain) for chain in pegasis_chains.values()]) * 0.001

    elif protocol == "leach":
        collision_prob = 0.4
        ch_receivers = dict()
        for node in nodes:
            if not node['alive']:
                continue
            packets_sent += 1
            if node['is_CH']:
                ch_id = node['id']
                dist = get_distance(node['pos'], bs_location)
                total_latency += 0.001
                energy_cost = (e_elec + e_amp * dist * dist) * packet_size
                node['energy'] = max(0, node['energy'] - energy_cost)
                if node['energy'] <= 0:
                    node['alive'] = False
                if random.random() > packet_loss_prob and (ch_receivers.get(ch_id, 0) <= 1 or random.random() > collision_prob):
                    packets_delivered += 1
            else:
                ch = next((ch for ch in cluster_heads if ch['id'] == node['assigned_to'] and ch['alive']), None)
                if ch:
                    ch_id = ch['id']
                    if ch_id not in ch_receivers:
                        ch_receivers[ch_id] = 0
                    ch_receivers[ch_id] += 1
                    dist = get_distance(node['pos'], ch['pos'])
                    total_latency += 0.001
                    energy_cost_tx = (e_elec + e_amp * dist * dist) * packet_size
                    energy_cost_rx = e_elec * packet_size
                    node['energy'] = max(0, node['energy'] - energy_cost_tx)
                    ch['energy'] = max(0, ch['energy'] - energy_cost_rx)
                    if node['energy'] <= 0:
                        node['alive'] = False
                    if ch['energy'] <= 0:
                        ch['alive'] = False
                    if random.random() > packet_loss_prob and (ch_receivers[ch_id] <= 1 or random.random() > collision_prob):
                        packets_delivered += 1
        for ch in cluster_heads:
            if ch['alive']:
                ch_id = ch['id']
                packets_sent += 1
                dist_to_bs = get_distance(ch['pos'], bs_location)
                energy_cost = (e_elec + e_amp * dist_to_bs * dist_to_bs) * packet_size
                ch['energy'] = max(0, ch['energy'] - energy_cost)
                if ch['energy'] <= 0:
                    ch['alive'] = False
                if random.random() > packet_loss_prob and (ch_receivers.get(ch_id, 0) <= 1 or random.random() > collision_prob):
                    packets_delivered += 1

    elif protocol == "pegasis":
        collision_prob = 0.4
        bs_receivers = 0
        chain = create_single_pegasis_chain(nodes)
        if not chain:
            return 0.0, 0.0
        total_latency = len(chain) * 0.001
        packets_sent += len(chain)
        for i in range(len(chain) - 1):
            sender_id = chain[i]
            receiver_id = chain[i + 1]
            sender = next(n for n in nodes if n['id'] == sender_id)
            receiver = next(n for n in nodes if n['id'] == receiver_id)
            dist = get_distance(sender['pos'], receiver['pos'])
            energy_cost_tx = (e_elec + e_amp * dist * dist) * packet_size
            energy_cost_rx = e_elec * packet_size
            sender['energy'] = max(0, sender['energy'] - energy_cost_tx)
            receiver['energy'] = max(0, receiver['energy'] - energy_cost_rx)
            if sender['energy'] <= 0:
                sender['alive'] = False
            if receiver['energy'] <= 0:
                receiver['alive'] = False
            if random.random() > packet_loss_prob:
                packets_delivered += 1
        last_id = chain[-1]
        last_node = next(n for n in nodes if n['id'] == last_id)
        dist_to_bs = get_distance(last_node['pos'], bs_location)
        energy_cost_to_bs = (e_elec + e_amp * dist_to_bs * dist_to_bs) * packet_size
        last_node['energy'] = max(0, last_node['energy'] - energy_cost_to_bs)
        if last_node['energy'] <= 0:
            last_node['alive'] = False
        bs_receivers += 1
        if random.random() > packet_loss_prob and (bs_receivers <= 1 or random.random() > collision_prob):
            packets_delivered += 1

    return total_latency, packets_delivered / packets_sent if packets_sent > 0 else 0

def print_simulation_results(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], round_num: int, protocol: str, latency: float, delivery_ratio: float):
    alive_nodes = [n for n in nodes if n['alive']]
    print(f"\nProtocol: {protocol.upper()}, Round {round_num}:")
    print("Total Alive Nodes:", len(alive_nodes))
    print("Cluster Heads:", [ch['id'] for ch in cluster_heads] if protocol != "pegasis" else "N/A")
    print("Average Energy:", round(sum(n['energy'] for n in alive_nodes) / len(alive_nodes) if alive_nodes else 0, 2))
    print("Average Latency (s):", round(latency, 4))
    print("Packet Delivery Ratio:", round(delivery_ratio, 4))
    if protocol == "hybrid-leap":
        print("\nPEGASIS Chains:")
        for ch_id, chain in pegasis_chains.items():
            print(f"CH {ch_id}: {chain}")
            print(f"Chain Length: {len(chain)}")
            chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain]
            avg_chain_energy = round(sum(n['energy'] for n in chain_nodes) / len(chain_nodes), 2) if chain_nodes else 0
            print(f"Average Energy in Chain: {avg_chain_energy}")

def plot_network(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], field_size: Tuple[int, int], bs_location: Tuple[int, int], save_plot: str, round_num: int, protocol: str):
    plt.figure(figsize=(10, 10))
    plt.title(f"{protocol.upper()} Network (Round {round_num})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    for node in nodes:
        if node['alive'] and not node['is_CH']:
            plt.plot(node['pos'][0], node['pos'][1], 'ko', markersize=7)
            plt.text(node['pos'][0] + 1, node['pos'][1] + 1, str(node['id']), fontsize=8)
    for node in nodes:
        if not node['alive']:
            plt.plot(node['pos'][0], node['pos'][1], 'kx', markersize=7)
    for ch in cluster_heads:
        if ch['alive']:
            plt.plot(ch['pos'][0], ch['pos'][1], 'r*', markersize=15)
            plt.text(ch['pos'][0] + 1, ch['pos'][1] + 1, str(ch['id']), fontsize=9, color='red')
    plt.plot(bs_location[0], bs_location[1], 'bs', markersize=14)
    if protocol == "hybrid-leap":
        colors = ['g', 'b', 'm', 'c', 'y']
        for idx, (ch_id, chain) in enumerate(pegasis_chains.items()):
            chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain if next((n for n in nodes if n['id'] == nid), None)['alive']]
            color = colors[idx % len(colors)]
            for i in range(len(chain_nodes) - 1):
                x_vals = [chain_nodes[i]['pos'][0], chain_nodes[i+1]['pos'][0]]
                y_vals = [chain_nodes[i]['pos'][1], chain_nodes[i+1]['pos'][1]]
                plt.plot(x_vals, y_vals, color + '-', linewidth=2)
    elif protocol == "pegasis":
        chain = create_single_pegasis_chain(nodes)
        chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain if next((n for n in nodes if n['id'] == nid), None)['alive']]
        for i in range(len(chain_nodes) - 1):
            x_vals = [chain_nodes[i]['pos'][0], chain_nodes[i+1]['pos'][0]]
            y_vals = [chain_nodes[i]['pos'][1], chain_nodes[i+1]['pos'][1]]
            plt.plot(x_vals, y_vals, 'g-', linewidth=2)
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
    plt.savefig(f"{save_plot}_{protocol}_round_{round_num}.png")
    plt.close()

def reset_ch_status(nodes: List[Dict]):
    """Reset CH status for new round."""
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
    num_rounds: int = 20,
    packet_loss_prob: float = 0.05,
    min_energy: float = 10.0,
    max_energy: float = 18.0,
    packet_size: int = 2000
):
    """Run the simulation for Hybrid-LEAP, LEACH, and PEGASIS."""
    set_random_seed(seed)
    protocols = ["hybrid-leap", "leach", "pegasis"]
    metrics = {protocol: {'residual_energy': [], 'alive_nodes': [], 'latency': [], 'delivery_ratio': []} for protocol in protocols}

    for protocol in protocols:
        nodes = create_nodes(num_nodes, field_size, min_energy, max_energy)
        for round_num in range(1, num_rounds + 1):
            reset_ch_status(nodes)
            cluster_heads = select_cluster_heads(nodes, ch_probability, bs_location, field_size, protocol, max_energy)
            pegasis_chains = {}
            if protocol == "hybrid-leap":
                assign_nodes_to_ch(nodes, cluster_heads)
                pegasis_chains = build_pegasis_chains(nodes, cluster_heads)
            elif protocol == "leach":
                assign_nodes_to_ch(nodes, cluster_heads)
            elif protocol == "pegasis":
                pegasis_chains = {0: create_single_pegasis_chain(nodes)}
            
            latency, delivery_ratio = simulate_transmission(nodes, protocol, cluster_heads, pegasis_chains, bs_location, packet_loss_prob, packet_size=packet_size)
            
            alive_nodes = sum(1 for n in nodes if n['alive'])
            total_energy = sum(n['energy'] for n in nodes if n['alive'])
            metrics[protocol]['residual_energy'].append(total_energy)
            metrics[protocol]['alive_nodes'].append(alive_nodes)
            metrics[protocol]['latency'].append(latency)
            metrics[protocol]['delivery_ratio'].append(delivery_ratio)
            
            print_simulation_results(nodes, cluster_heads, pegasis_chains, round_num, protocol, latency, delivery_ratio)
            plot_network(nodes, cluster_heads, pegasis_chains, field_size, bs_location, save_plot, round_num, protocol)

    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(['residual_energy', 'alive_nodes', 'latency', 'delivery_ratio'], 1):
        plt.subplot(2, 2, i)
        for protocol in protocols:
            plt.plot(metrics[protocol][metric], label=protocol.upper())
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel("Round")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_plot}_all_metrics_overall.png")
    plt.close()

    data = []
    for protocol in protocols:
        for round_num in range(len(metrics[protocol]['residual_energy'])):
            data.append({
                'Protocol': protocol.upper(),
                'Round': round_num + 1,
                'Residual Energy': metrics[protocol]['residual_energy'][round_num],
                'Alive Nodes': metrics[protocol]['alive_nodes'][round_num],
                'Latency (s)': metrics[protocol]['latency'][round_num],
                'Delivery Ratio': metrics[protocol]['delivery_ratio'][round_num]
            })
    df = pd.DataFrame(data)
    df.to_csv(f"{save_plot}_results.csv", index=False)
    print(f"Simulation results exported to {save_plot}_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WSN simulation with Hybrid-LEAP, LEACH, and PEGASIS.")
    parser.add_argument('--num_nodes', type=int, default=75, help="Number of nodes")
    parser.add_argument('--field_size', type=int, nargs=2, default=[400, 400], help="Field size (width height)")
    parser.add_argument('--ch_probability', type=float, default=0.1, help="Cluster head probability")
    parser.add_argument('--bs_location', type=int, nargs=2, default=[200, 200], help="Base station location (x y)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed (optional)")
    parser.add_argument('--save_plot', type=str, default='plot', help="Filename prefix to save the plots")
    parser.add_argument('--num_rounds', type=int, default=100, help="Number of simulation rounds")
    parser.add_argument('--packet_loss_prob', type=float, default=0.1, help="Packet loss probability per hop")
    parser.add_argument('--min_energy', type=float, default=0.1, help="Minimum initial energy for nodes (Joules)")
    parser.add_argument('--max_energy', type=float, default=1.0, help="Maximum initial energy for nodes (Joules)")
    parser.add_argument('--packet_size', type=int, default=4000, help="Packet size in bits")
    args = parser.parse_args()

    run_simulation(
        num_nodes=args.num_nodes,
        field_size=tuple(args.field_size),
        ch_probability=args.ch_probability,
        bs_location=tuple(args.bs_location),
        seed=args.seed,
        save_plot=args.save_plot,
        num_rounds=args.num_rounds,
        packet_loss_prob=args.packet_loss_prob,
        min_energy=args.min_energy,
        max_energy=args.max_energy,
        packet_size=args.packet_size
    )