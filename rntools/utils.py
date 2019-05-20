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
    edge['capacity'] = max_speed * lanes / 3.6

# Sample a demand
def sample_demand(graph, num_od_pairs=10, rate_range=(0,1)):
    demand = np.zeros((num_od_pairs, 3))
    # Sample od-pairs
    nodes = np.fromiter(graph.nodes(), dtype=int)
    N = len(nodes)
    sampled_ind = np.random.randint(N,size=(num_od_pairs, 2))
    sampled_ods = nodes[sampled_ind]
    # Sample demand rate
    sampled_rates = np.random.uniform(*rate_range, num_od_pairs)
    return (sampled_ods[:, 0], sampled_ods[:, 1], sampled_rates)
