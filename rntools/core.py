import utils as ut
import visualization

class RNTools:
    def __init__(self, G, demand):
        self.G = G
        self.solution_flow = None
        self.demand = demand

        if demand is None:
            self.demand = self._

    def add_cost_and_capacity_metric(self):
        for e in self.G.edges:
            utils.add_cost_metric(self.G.edges[e])
            utils.add_capacity_metric(self.G.edges[e])


    # Construct the linear problem
    def construct_LP(self, demand, lbda=0):
        """
        lbda is the weight of L1 regularization of flow
        """
        N = len(self.G.nodes)
        M = len(self.G.edges)
        num_commodities = demand.num_commodities

        # Define the MCF problem
        capacity = np.random.uniform(10, size=(M,))
        cost = np.random.uniform(10, size=(M,))
        origs, dests, rates = sample_demand(G, num_od_pairs=num_commodities)
        node_to_ind = {n:i for (i,n) in enumerate(G.nodes)}

        # lbda is the weight of L1-regularization
        lbda = 5

        # Build the program
        incidence_matrix = nx.incidence_matrix(G, oriented=True)
        constraints = []
        objective = 0
        for i in range(num_commodities):
            flow = cp.Variable(shape=M)
            node_flow = np.zeros(N)
            node_flow[node_to_ind[origs[i]]] = -rates[i]
            node_flow[node_to_ind[dests[i]]] = rates[i]
            constraints += [incidence_matrix * flow == node_flow,
                           flow >= 0,
                           flow <= capacity]
            objective += cost.T * flow + lbda * cp.sum(flow)

        # Solve the program
        objective = cp.Minimize(objective)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='GUROBI')

        # Compile the answers
        total_flow = np.zeros(M)
        for v in prob.variables():
            total_flow = total_flow + v.value
        self.solution_flow = total_flow

    # Visualization
    def nx_draw(self, edge_color=None, with_labels=True):
        nx.draw_circular(self.G, edge_color=edge_color, with_labels=with_labels)

    def ox_draw(self, edge_color=None, node_color=None):
        visualization.plot_graph(self.G, ec=edge_color, nc=node_color)
