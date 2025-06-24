import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

def set_random_seed(seed: Optional[int] = None):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)

def create_nodes(num_nodes: int, field_size: Tuple[int, int]) -> List[Dict]:
    """Create nodes with random positions and initial energy."""
    nodes = []
    for i in range(num_nodes):
        node = {
            'id': i,
            'pos': (random.uniform(0, field_size[0]), random.uniform(0, field_size[1])),
            'energy': round(1.0 + 0.2 * (i % 5), 2),
            'is_CH': False,
            'assigned_to': None
        }
        nodes.append(node)
    return nodes

def get_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def select_cluster_heads(nodes: List[Dict], ch_probability: float) -> List[Dict]:
    """Select cluster heads based on probability."""
    cluster_heads = []
    for node in nodes:
        if random.random() < ch_probability:
            node['is_CH'] = True
            cluster_heads.append(node)
    return cluster_heads

def assign_nodes_to_ch(nodes: List[Dict], cluster_heads: List[Dict]):
    """Assign each non-CH node to the nearest cluster head."""
    for node in nodes:
        if not node['is_CH'] and cluster_heads:
            closest_ch = min(cluster_heads, key=lambda ch: get_distance(node['pos'], ch['pos']))
            node['assigned_to'] = closest_ch['id']

def create_pegasis_chain(members: List[Dict]) -> List[Dict]:
    """Form a PEGASIS chain among the given members."""
    if not members:
        return []
    chain = [members[0]]
    visited = {members[0]['id']}
    while len(chain) < len(members):
        last = chain[-1]
        next_node = min(
            [n for n in members if n['id'] not in visited],
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
        members = [n for n in nodes if n['assigned_to'] == ch['id']]
        full_group = [ch] + members
        chain = create_pegasis_chain(full_group)
        pegasis_chains[ch['id']] = [n['id'] for n in chain]
    return pegasis_chains

def print_simulation_results(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]]):
    print("Total Nodes:", len(nodes))
    print("Cluster Heads:", [ch['id'] for ch in cluster_heads])
    print("Average Energy:", round(sum(n['energy'] for n in nodes) / len(nodes), 2))
    print("\nPEGASIS Chains:")
    for ch_id, chain in pegasis_chains.items():
        print(f"CH {ch_id}: {chain}")

def plot_network(nodes: List[Dict], cluster_heads: List[Dict], pegasis_chains: Dict[int, List[int]], field_size: Tuple[int, int], bs_location: Tuple[int, int]):
    plt.figure(figsize=(10, 10))
    plt.title("Hybrid-LEAP Node Deployment with CHs and PEGASIS Chains")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # Plot all normal nodes
    for node in nodes:
        if not node['is_CH']:
            plt.plot(node['pos'][0], node['pos'][1], 'ko', markersize=7)
            plt.text(node['pos'][0] + 1, node['pos'][1] + 1, str(node['id']), fontsize=8)
    # Plot cluster heads
    for ch in cluster_heads:
        plt.plot(ch['pos'][0], ch['pos'][1], 'r*', markersize=15)
        plt.text(ch['pos'][0] + 1, ch['pos'][1] + 1, str(ch['id']), fontsize=9, color='red')
    # Plot base station
    plt.plot(bs_location[0], bs_location[1], 'bs', markersize=14)
    # Plot PEGASIS chains with different colors
    colors = ['g', 'b', 'm', 'c', 'y']
    for idx, (ch_id, chain) in enumerate(pegasis_chains.items()):
        chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain]
        color = colors[idx % len(colors)]
        for i in range(len(chain_nodes) - 1):
            x_vals = [chain_nodes[i]['pos'][0], chain_nodes[i+1]['pos'][0]]
            y_vals = [chain_nodes[i]['pos'][1], chain_nodes[i+1]['pos'][1]]
            plt.plot(x_vals, y_vals, color+'-', linewidth=2)
    # Custom legend with correct markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Normal Node'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=15, label='Cluster Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='b', markersize=12, label='Base Station'),
        Line2D([0], [0], color='g', lw=2, label='PEGASIS Link')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.xlim(0, field_size[0] + 10)
    plt.ylim(0, field_size[1] + 10)
    plt.grid(True)
    plt.show()

def run_simulation(
    num_nodes: int = 20,
    field_size: Tuple[int, int] = (100, 100),
    ch_probability: float = 0.2,
    bs_location: Tuple[int, int] = (100, 100),
    seed: Optional[int] = None
):
    """Run the Hybrid-LEAP simulation with PEGASIS chains."""
    set_random_seed(seed)
    nodes = create_nodes(num_nodes, field_size)
    cluster_heads = select_cluster_heads(nodes, ch_probability)
    assign_nodes_to_ch(nodes, cluster_heads)
    pegasis_chains = build_pegasis_chains(nodes, cluster_heads)
    print_simulation_results(nodes, cluster_heads, pegasis_chains)
    plot_network(nodes, cluster_heads, pegasis_chains, field_size, bs_location)

if __name__ == "__main__":
    # You can change parameters here
    run_simulation(
        num_nodes=20,
        field_size=(100, 100),
        ch_probability=0.2,
        bs_location=(100, 100),
        seed=None  # Set to None for random results
    )
