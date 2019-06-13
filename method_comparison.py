from rntools.demand import Demand as Demand
from rntools.utils import sample_demand
import rntools.visualization
import rntools.core
import rntools.io
import rntools.utils
import numpy as np

#-----------------PARAMS-------------------------
experiment_name = 'Kamppi'

amodeus_data = False
graph_path  = './data/Kamppi.graphml'
demand_path = './data/SF_demand.xml'

lbda        = 0
phi         = 0
num_samples = 1

resample    = 1
exp_num_demands = [1, 2, 3]
#-----------------SCRIPT-------------------------

def compare_result(graph, demand):
    result = np.zeros(3)
    # LP formulation
    lp = rntools.core.LPSparsifier(graph, demand, lbda=1)
    lp.solve()
    result[0] = lp.num_useful_edges
    # Weighted LP formulation
    wlp = rntools.core.WeightedLPSparsifier(graph, demand, lbda=5)
    wlp.solve()
    result[1] = wlp.num_useful_edges
    # Bisection for actual minimum
    bisec = rntools.core.MILPBisectionSparsifier(graph, demand, np.inf)
    bisec.solve()
    result[2] = bisec.num_useful_edges
    return result

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

# Run the experiment
rn = load_graph()
results = np.zeros( (len(exp_num_demands), 3) )
for _ in range(resample):
    for i, N in enumerate(exp_num_demands):
        subdemand = sample_exp_demand(rn, N)
        results[i] = results[i] + compare_result(rn, subdemand)
results = results / resample

# Plot the results
lp_res = results[:, 0]
wlp_res = results[:, 1]
bis_res = results[:, 2]

ind = np.arange(len(lp_res))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, lp_res, width, label='LP')
rects2 = ax.bar(ind, wlp_res, width, label='Weighted LP')
rects3 = ax.bar(ind + width, bis_res, width, label='Oracle')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of edges')
ax.set_title('Comparison across methods for num edges')
ax.set_xticks(ind)
ax.set_xticklabels(('Low Demand', 'Medium Demand', 'High Demand'))
ax.legend()
