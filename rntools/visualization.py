
# Plotting the graph
def edge_color_from_attribute(graph, attr):
    values = [v[attr] for _,_,v in graph.edges(data=True)]
    maxima = max(values)
    minima = min(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Greys')
    ec = [mapper.to_rgba(v) for v in values]
    return ec
    
def plot_graph(graph, ec=None, ec_key=None, nc=None):
    if not ec:
        ec = ['grey' for edge in graph.edges()]
    if ec_key:
        ec = ox.get_edge_colors_by_attr(graph, attr=ec_key)
    if not nc:
        nc = ['white' for node in graph.nodes()]
    fig, ax = ox.plot_graph(graph, edge_color=ec, node_color=nc)
    plt.tight_layout()

def node_color_from_demand(graph, demand):
    origs, dests, rates = demand

    node_to_rate = dict()
    for o,d,r in zip(origs, dests, rates):
        node_to_rate[o] = r
        node_to_rate[d] = r

    values = rates
    maxima = max(values)
    minima = min(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Oranges')

    nc = [(1,1,1) for _ in graph.nodes()]
    for (i, node) in enumerate( graph.nodes() ):
        if node in node_to_rate.keys():
            nc[i] = mapper.to_rgba(node_to_rate[node])
    return nc
