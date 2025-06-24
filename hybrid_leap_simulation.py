
import random
import math
import matplotlib.pyplot as plt

# إعدادات الشبكة
NUM_NODES = 20
FIELD_SIZE = (100, 100)
CH_PROBABILITY = 0.2
BS_LOCATION = (100, 100)

# إنشاء العقد بخصائص عشوائية
nodes = []
for i in range(NUM_NODES):
    node = {
        'id': i,
        'pos': (random.uniform(0, FIELD_SIZE[0]), random.uniform(0, FIELD_SIZE[1])),
        'energy': round(1.0 + 0.2 * (i % 5), 2),
        'is_CH': False,
        'assigned_to': None
    }
    nodes.append(node)

# دالة لحساب المسافة بين نقطتين
def get_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# اختيار رؤساء المجموعات باستخدام LEACH
cluster_heads = []
for node in nodes:
    if random.random() < CH_PROBABILITY:
        node['is_CH'] = True
        cluster_heads.append(node)

# تعيين كل عقدة إلى أقرب CH
for node in nodes:
    if not node['is_CH'] and cluster_heads:
        closest_ch = min(cluster_heads, key=lambda ch: get_distance(node['pos'], ch['pos']))
        node['assigned_to'] = closest_ch['id']

# تشكيل سلسلة PEGASIS لكل مجموعة
def create_pegasis_chain(members):
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

# إنشاء السلاسل لكل CH
pegasis_chains = {}
for ch in cluster_heads:
    members = [n for n in nodes if n['assigned_to'] == ch['id']]
    full_group = [ch] + members
    chain = create_pegasis_chain(full_group)
    pegasis_chains[ch['id']] = [n['id'] for n in chain]

# طباعة النتائج
print("Total Nodes:", NUM_NODES)
print("Cluster Heads:", [ch['id'] for ch in cluster_heads])
print("Average Energy:", round(sum(n['energy'] for n in nodes) / NUM_NODES, 2))
print("\nPEGASIS Chains:")
for ch_id, chain in pegasis_chains.items():
    print(f"CH {ch_id}: {chain}")

# رسم الشبكة والعقد
plt.figure(figsize=(10, 10))
plt.title("Hybrid-LEAP Node Deployment with CHs and PEGASIS Chains")
plt.xlabel("X Position")
plt.ylabel("Y Position")

# رسم جميع العقد
for node in nodes:
    plt.plot(node['pos'][0], node['pos'][1], 'ko')  # العقد العادية
    plt.text(node['pos'][0] + 1, node['pos'][1] + 1, str(node['id']), fontsize=8)

# رسم رؤساء المجموعات
for ch in cluster_heads:
    plt.plot(ch['pos'][0], ch['pos'][1], 'ro', markersize=10)

# رسم محطة القاعدة
plt.plot(BS_LOCATION[0], BS_LOCATION[1], 'bs', markersize=12, label="Base Station")

# رسم خطوط PEGASIS لكل مجموعة
for ch_id, chain in pegasis_chains.items():
    chain_nodes = [next(n for n in nodes if n['id'] == nid) for nid in chain]
    for i in range(len(chain_nodes) - 1):
        x_vals = [chain_nodes[i]['pos'][0], chain_nodes[i+1]['pos'][0]]
        y_vals = [chain_nodes[i]['pos'][1], chain_nodes[i+1]['pos'][1]]
        plt.plot(x_vals, y_vals, 'g--', linewidth=1)

plt.legend(['Normal Node', 'Cluster Head', 'Base Station', 'PEGASIS Link'], loc='upper left')
plt.grid(True)
plt.show()
