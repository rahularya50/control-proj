import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, bin_size, xlim, nominal_value=None):
    hist, edges = np.histogram(data, bin_size)
    edge_means = [(edges[i] + edges[i+1])/2 for i in range(len(edges) -1)]
    fig, ax = plt.subplots()
    ax.plot(edge_means, hist)
    if nominal_value is not None:
        line_height = max(hist)
        ax.plot([nominal_value] * line_height, range(line_height))
        ax.set_xlim(left=xlim[0], right=xlim[1])