import numpy as np
from matplotlib import pyplot as plt

results = np.loadtxt('./method_Kamppi_raw.txt')

# Plot the results
lp_res = results[:, 0]
wlp_res = results[:, 1]
bis_res = results[:, 2]

ind = np.arange(len(lp_res))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, lp_res, width, label='LP')
rects2 = ax.bar(ind, wlp_res, width, label='L1 reg')
rects3 = ax.bar(ind + width, bis_res, width, label='Weighted L1 reg')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of edges')
ax.set_title('Comparison across methods for num edges (Kamppi)')
ax.set_xticks(ind)
ax.set_xticklabels(('5 Demands', '15 Demands', '30 Demands'))
ax.legend()

plt.show()
