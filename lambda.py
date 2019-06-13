from rntools.demand import Demand as Demand
from rntools.utils import sample_demand
from matplotlib import pyplot as plt
import rntools.visualization
import rntools.core
import rntools.io
import rntools.utils
import numpy as np
import json

#-----------------PARAMS-------------------------
experiment_name = 'Kamppi_Lambda'

amodeus_data = False
graph_path  = './data/Kamppi.graphml'
demand_path = './data/SF_demand.xml'

lbdas    = [pow(10, i) for i in range(5)]
phi         = 0
num_samples = 1

resample    = 2
exp_num_demands = 1
#-----------------SCRIPT-------------------------

def load_graph():
    if amodeus_data:
        rn = rntools.io.read_MATSim_network(graph_path)
    else:
        rn = rntools.io.load_graph(graph_path)
        rntools.utils.add_cost_and_capacity_metric(rn)
    return rn

def sample_exp_demand(rn, num_demand):
    if amodeus_data:
        demand = rntools.io.read_MATSim_demand(demand_path, rn)
        subdemand = rntools.utils.sample_subdemand(demand, num_demand)
    else:
        origs, dests, rates = sample_demand(rn, num_od_pairs=num_demand,
                                            rate_range=(1,5))
        subdemand = Demand(origs, dests, rates)
    return subdemand

def compute_lambda_trend(graph, demand):
    baseline_edges, baseline_costs = [], []
    lp_edges, lp_costs = [], []
    wlp_edges, wlp_costs = [], []

    for l in lbdas:
        bl = rntools.core.LPSparsifier(graph, demand, lbda=0); bl.solve()
        lp = rntools.core.LPSparsifier(graph, demand, lbda=l); lp.solve()
        wlp = rntools.core.WeightedLPSparsifier(graph, demand, lbda=l); wlp.solve()
        baseline_edges.append(bl.num_useful_edges); baseline_costs.append(bl.compute_cost())
        lp_edges.append(lp.num_useful_edges); lp_costs.append(lp.compute_cost())
        wlp_edges.append(wlp.num_useful_edges); wlp_costs.append(wlp.compute_cost())

    return baseline_edges, baseline_costs, lp_edges, lp_costs, wlp_edges, wlp_costs

# Run the experiment
rn = load_graph()
be, bc, le, lc, we, wc = [], [], [], [], [], []
for _ in range(resample):
    demand = sample_exp_demand(rn, exp_num_demands)
    res = compute_lambda_trend(rn, demand)
    be.append(res[0])
    bc.append(res[1])
    le.append(res[2])
    lc.append(res[3])
    we.append(res[4])
    wc.append(res[5])

# Plot the scatter of num_edges vs cost
edges = be + le + we
costs = bc + lc + wc
np.savetxt('./results/'+experiment_name+'_num_edges', edges)
np.savetxt('./results/'+experiment_name+'_costs', costs)
plt.scatter(le, lc, label='LP')
plt.scatter(we, wc, label='weighted LP')
plt.scatter(be, bc, label='baseline')
plt.legend()
plt.show()

# Plot the changes in num_edges vs lambda
be = np.array(be).mean()
le = np.array(le).mean(0)
we = np.array(we).mean(0)
ll, = plt.plot(lbdas, le)
lw, = plt.plot(lbdas, we)
plt.legend([ll, lw], ['LP', 'WLP'])
plt.show()
