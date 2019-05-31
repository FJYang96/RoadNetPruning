from rntools.demand import Demand as Demand
import rntools.visualization
import rntools.core
import rntools.io
from rntools.utils import sample_demand
import rntools.utils
import numpy as np

#-----------------PARAMS-------------------------
graph_path  = './data/SF.xml'
demand_path = './data/SF_demand.xml'
num_demand  = 1
lbda        = 0
phi         = 0
num_samples = 1

#-----------------SCRIPT-------------------------
rn = rntools.io.read_MATSim_network(graph_path)
demand = rntools.io.read_MATSim_demand(demand_path, rn)
subdemand = rntools.utils.sample_subdemand(demand, num_demand)

problem = rntools.core.RNTools(rn, subdemand)
problem.construct_LP(lbda=lbda, phi=phi, num_samples=num_samples)
problem.solve()

np.savetxt('results/solution.txt', problem.solution_flow)
problem.ox_draw(save=True, filename='lol')
