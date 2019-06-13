import rntools.visualization as vis
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import cvxpy as cp
import osmnx as ox

class Sparsifier:
    '''
    A wrapper for solving a graph sparsification problem.
    This is a (somewhat) abstract class, and should be instantiated to specific methods
    Visualization and I/O, however, are implemented here as they are universal
    Different methods have different solve() methods
    '''
    def __init__(self, G, demand):
        self.G = G
        self.demand = demand
        self.solved = False
        self.num_useful_edges = np.inf
        self.solution = None

    def solve(self):
        raise NotImplementedError

    def nx_draw(self):
        vis.plot_nx_solution(self.G, self.solution, self.demand)

    def ox_draw(self, edge_color=None, node_color=None,
                save=False, filename='temp'):
        vis.plot_ox_solution(self.G, self.solution, self.demand,
                                       save=save, filename=filename)

    def write_flow_to_file(self, filename):
        np.savetxt(filename, self.solution)

    def load_flow_from_file(self, filename):
        self.solution = np.loadtxt(filename)
        self.num_useful_edges = len(np.nonzero(self.solution)[0])

class OptimizationSparsifier(Sparsifier):
    '''
    A class of sparsifiers that finds useful edges by solving an optimization
    problem
    This will be further instantiated into LP formulation and combinatorial 
    formulation
    '''
    def __init__(self, G, demand):
        super().__init__(G, demand)
        self.prob = None
        self.objective = 0
        self.constraints = []
        self.add_variables()

    # Building blocks for the program

    def add_variables(self):
        '''
        Add variables that are helpful to constructing the opt problem
        '''
        # Quantities that are useful in building the program
        self.N = len(self.G.nodes)
        self.M = len(self.G.edges)
        self.cost = np.array([self.G.edges[e]['weight'] \
                                  for e in self.G.edges])
        self.max_cost = np.max(self.cost)
        self.capacity = np.array([self.G.edges[e]['capacity'] \
                                  for e in self.G.edges])
        self.node_to_ind = {n:i for (i,n) in enumerate(self.G.nodes)}
        self.total_flow = np.zeros(self.M)

        # Define the variables for the problem
        self.flows = []
        self.solution = np.zeros(self.M)
        for _ in range(self.demand.num_commodities):
            fi = cp.Variable(shape=self.M) 
            self.flows.append(fi) 
            self.solution = self.solution + fi
            
    def construct_program(self):
        '''
        Construct the optimization problem.
        '''
        raise NotImplementedError

    def solve(self, verbose=False):
        raise NotImplementedError

class LPSparsifier(OptimizationSparsifier):
    def __init__(self, G, demand, lbda, phi=0, num_samples=24):
        super().__init__(G, demand)
        self.lbda = lbda
        self.phi = phi
        self.num_samples = num_samples

    def construct_program(self):
        """
        lbda is the weight of L1 regularization of flow
        """
        self.add_flow_constraints(phi=self.phi, num_samples=self.num_samples)
        self.objective += self.cost.T * self.solution / self.max_cost + \
            self.lbda * cp.sum(self.solution)
        self.prob = cp.Problem(cp.Minimize(self.objective), self.constraints)

    def add_flow_constraints(self, phi=0, num_samples=24):
        '''
        Nonnegative, less than capacity, continuous
        '''
        incidence_matrix = nx.incidence_matrix(self.G, oriented=True)
        for i in range(self.demand.num_commodities):
            node_flow = np.zeros(self.N)
            rate = self.demand.rates[i]
            # augment the flow rates to encourage robust solutions
            robust_rate = rate + phi * np.sqrt(rate / num_samples)
            node_flow[self.node_to_ind[self.demand.origs[i]]] = -robust_rate
            node_flow[self.node_to_ind[self.demand.dests[i]]] += robust_rate
            self.constraints += [incidence_matrix * self.flows[i] == node_flow,
                                 self.flows[i] >= 0]
        self.constraints += [self.solution <= self.capacity]

    def solve(self, verbose=False, solver_verbose=False):
        ''' Solve the LP '''
        self.construct_program()
        self.prob.solve(solver='GUROBI', verbose=solver_verbose)
        if self.objective.value is None or np.isinf(self.objective.value):
            print('Sparsification Failed')
            print('Cannot route the demand')
            return False

        # Compile the answers
        self.solution = self.solution.value
        self.num_useful_edges = len(np.nonzero(self.solution)[0])
        self.solved=True
        if verbose:
            print('Sparsification Successful')
            print('The graph has',self.num_useful_edges,'edges')
        return True

    def compute_cost(self):
        if self.solved:
            return np.dot(self.solution, self.cost)
        else:
            return np.inf

class WeightedLPSparsifier(LPSparsifier):
    '''
    Basically identical to LP Sparsifier; However, here the regularizer weights 
    the edges inversely by their capacity
    '''
    def construct_program(self):
        """
        lbda is the weight of L1 regularization of flow
        """
        self.add_flow_constraints(phi=self.phi, num_samples=self.num_samples)
        self.objective += self.cost.T * self.solution / self.max_cost + \
            self.lbda * cp.sum(self.solution / self.capacity) * np.mean(self.capacity)
        self.prob = cp.Problem(cp.Minimize(self.objective), self.constraints)

class MILPSparsifier(LPSparsifier):
    def __init__(self, G, demand, lbda, budget, phi=0, num_samples=24):
        super().__init__(G, demand, lbda, phi, num_samples)
        self.budget = budget

    def construct_program(self):
        """
        lbda is the weight of L1 regularization of flow
        """
        self.add_flow_constraints(phi=self.phi, num_samples=self.num_samples)
        self.add_budget_constraints(self.budget)
        self.objective += self.cost.T * self.solution + \
            self.lbda * cp.sum(self.solution / self.capacity)
        self.prob = cp.Problem(cp.Minimize(self.objective), self.constraints)

    def add_budget_constraints(self, budget):
        '''
        Combinatorial constraint on how many edges can be active
        '''
        self.mask = cp.Variable(shape=self.M, boolean=True)
        self.constraints += [cp.sum(self.mask) <= budget,
                             self.solution <= cp.multiply(self.mask,
                                                          self.capacity)]

    def solve(self, verbose=False, solver_verbose=False):
        ''' Solve the LP '''
        self.construct_program()
        self.prob.solve(solver='GUROBI', verbose=solver_verbose)
        if self.objective.value is None or np.isinf(self.objective.value):
            print('Sparsification Failed')
            print('Demand cannot be routed under current budget')
            return False

        # Compile the answers
        self.solved=True
        self.mask = self.mask.project(self.mask.value)
        self.num_useful_edges = np.sum(self.mask)
        self.solution = self.solution.value
        self.solution[self.mask == 0] = 0
        if verbose:
            print('Sparsification Successful')
            print('The graph has',self.num_useful_edges,'edges')
        return True

class MILPBisectionSparsifier(Sparsifier):
    def __init__(self, G, demand, max_cost, phi=0, num_samples=24, precision=1):
        self.G = G
        self.demand = demand
        self.max_cost = max_cost
        self.phi = phi
        self.num_samples = num_samples
        self.precision = precision

    def construct_program(self):
        pass

    def solve(self):
        M_low, M_high = 0, len(self.G.edges)
        while( np.abs(M_low - M_high) > self.precision ):
            edge_budget = int( (M_low + M_high) / 2 )
            program = MILPSparsifier(self.G, self.demand, 0, edge_budget,
                                     self.phi, self.num_samples)
            if program.solve():
                self.program = program
                self.solution = program.solution
                self.num_useful_edges = program.num_useful_edges
            if program.compute_cost() >= self.max_cost:
                M_low = edge_budget
            else:
                M_high = edge_budget

    def compute_cost(self):
        return self.program.compute_cost()
