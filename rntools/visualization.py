from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm
import networkx as nx
import osmnx as ox

# Plotting the graph
def edge_color_from_attribute(graph, attr):
    values = [v[attr] for _,_,v in graph.edges(data=True)]
    maxima = max(values)
    minima = min(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Greys')
    ec = [mapper.to_rgba(v) for v in values]
    return ec

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

def plot_nx_solution(graph, solution_flow, demand=None,
                     save=False, filename='temp'):
    '''
    Plot the solution flow for small toy networks in networkx
    If demand is not None, plot the demand on the left, and the solution
    on the right. Otherwise, just plot the solution flow
    '''
    if demand is not None:
        # Plot demands
        plt.subplot(1,2,1)
        od_pairs = zip(demand.origs, demand.dests)
        edgelist = [od for od in od_pairs]
        nx.draw_circular(graph, edgelist=edgelist, with_labels=True)
        # Plot the useful edges
        plt.subplot(1,2,2)
    edgelist = [e for (i,e) in enumerate(graph.edges) \
                 if solution_flow[i] > 0]
    nx.draw_circular(graph, edgelist=edgelist, with_labels=True)

def plot_ox_solution(graph, solution_flow, demand=None,
                     save=False, filename='temp'):
    '''
    Plot the solution flow on top of the road network by highlighting the
    useful edges in red
    '''
    # Use node color to symbolize demand
    node_color = ['white' for n in graph.nodes]
    if demand is not None:
        od_pairs = zip(demand.origs, demand.dests)
        node_color = ['blue' if n in demand.origs or n in demand.dests \
                      else 'white' for n in graph.nodes]
    # Use edge color to highlight edges that are useful
    edge_color = ['red' if solution_flow[i] > 0 else 'lightgrey' \
                  for (i,e) in enumerate(graph.edges)]
    # Plot the graph
    fig, ax = ox.plot_graph(graph, edge_color=edge_color, node_color=node_color,
                            save=save, filename=filename)
    plt.tight_layout()
