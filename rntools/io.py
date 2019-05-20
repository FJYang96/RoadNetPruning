import xml.etree.ElementTree as ET
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

# Download a road network from Open Street Map
def download_road_network(place_query):
    '''
    Place query can be a dictionary or a string, for example:
        place_query = {'city':'San Francisco', 'state':'California', 'country':'USA'}, or
        place_query = "Kamppi, Helsinki, Finland"
    '''
    return ox.graph_from_place(place_query, network_type='drive')

# File I/O
def save_graph(graph, directory):
    ox.save_graphml(graph, directory, folder='.')
def load_graph(directory):
    return ox.load_graphml(directory, folder='.')

def read_MATSim_network(directory):
    # Read MATSim road graph
    tree = ET.parse(directory)
    root = tree.getroot()
    nodes = root[0]
    edges = root[1]

    G = nx.MultiDiGraph()

    def get_node_coord(node):
        return (node['x'], node['y'])

    def get_line_between_nodes(n1, n2):
        coord1 = get_node_coord(n1)
        coord2 = get_node_coord(n2)
        return LineString([coord1, coord2])

    for node in nodes:
        n = node.attrib
        node_id = int(n['id'])
        node_x = float(n['x'])
        node_y = float(n['y'])
        G.add_node(node_id, x=node_x, y=node_y, osmid=node_id)

    for edge in edges:
        
        e = edge.attrib
        
        # Convert the variables into correct types
        e['id'] = int(e['id'])
        e['from'] = int(e['from'])
        e['to'] = int(e['to'])
        e['length'] = float(e['length'])
        e['freespeed'] = float(e['freespeed'])
        e['capacity'] = float(e['capacity'])
        e['permlanes'] = float(e['permlanes'])
        e['oneway'] = int(e['oneway'])
        
        # Add a shape field for plotting. Shape is assumed to be a straight 
        # line connecting the end points
        start = float(e['from'])
        end = float(e['to'])
        e['shape'] = get_line_between_nodes(G.node[start], G.node[end])
        
        # Add the edge to the graph
        G.add_edge(start, end, key=0, **e)

    G.graph['crs'] = {'datum': 'WGS84',
      'ellps': 'WGS84',
      'proj': 'utm',
      'zone': 35,
      'units': 'm'}
    G.graph['name'] = 'SF'
    
    return G
