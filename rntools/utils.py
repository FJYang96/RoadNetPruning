import numpy as np
import rntools.demand

# Convert a OSM MultiDiGraph to simple graph
def reduce_to_simple(graph):
    proj_graph = ox.project_graph(graph)
    simple = nx.Graph( proj_graph.to_undirected() )
    return simple

# Convert a simple graph back to a MultiDiGraph
def simple_to_multi(simple):
    return nx.MultiGraph(simple)

# Add cost & capacity to a simple graph
def add_cost_metric(edge):
    max_speed = 40.0
    if 'maxspeed' in edge.keys():
        max_speed = np.fromiter(edge['maxspeed'], float).mean()
    length = edge['length']
    edge['cost'] = length / max_speed * 3.6 #(km/h -> m/s)

def add_capacity_metric(edge):
    max_speed = 40.0
    if 'maxspeed' in edge.keys():
        max_speed = np.fromiter(edge['maxspeed'], float).mean()
    lanes = 1.0
    if 'lanes' in edge.keys():
        lanes = np.fromiter(edge['lanes'], float).sum()
    edge['capacity'] = max_speed * lanes / 3.6 * 1000

def add_cost_and_capacity_metric(graph):
    for e in graph.edges:
        add_cost_metric(graph.edges[e])
        add_capacity_metric(graph.edges[e])

# Sample a demand
def sample_demand(graph, num_od_pairs=10, rate_range=(0,1)):
    demand = np.zeros((num_od_pairs, 3))
    # Sample od-pairs
    nodes = np.fromiter(graph.nodes(), dtype=int)
    N = len(nodes)
    # Sample indices; prevent origin = destination
    sampled_ind = np.random.randint(N,size=(num_od_pairs, 2))
    for i in range(num_od_pairs):
        if sampled_ind[i, 0] == sampled_ind[i, 1]:
            sampled_ind[i, 1] = (sampled_ind[i, 1] + 1) % N
    sampled_ods = nodes[sampled_ind]
    # Sample demand rate
    sampled_rates = np.random.uniform(*rate_range, num_od_pairs)
    return (sampled_ods[:, 0], sampled_ods[:, 1], sampled_rates)

def sample_subdemand(demand, N):
    ind = np.random.choice(demand.num_commodities, size=N, replace=False)
    sampled_origs = np.array(demand.origs)[ind]
    sampled_dests = np.array(demand.dests)[ind]
    sampled_rates = np.array(demand.rates)[ind]
    return rntools.demand.Demand(sampled_origs, sampled_dests, sampled_rates)
