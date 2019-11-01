import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

np.random.seed()

def trials(a, u, w_param, x0_param, N, T):
    mu_x, sigma_x = x0_param
    mu_w, sigma_w = w_param
    np.random.seed()
    w_trace = []
    x_trace = [np.random.normal(loc=mu_x, scale=sigma_x, size=N)]
    a_hat_trace = [np.zeros(N)]
    y_trace = [np.zeros(N)]
    a_vec = np.repeat(a, N)
    u_vec = np.repeat(u, N)
    
    for i in range(T-1):
        # Get current state
        x_t = x_trace[-1]
        a_hat_t = a_hat_trace[-1]
        y_t = y_trace[-1]
        w_t = np.random.normal(loc=mu_w, scale=sigma_w, size=N)
        
        # State update
        x_new = np.multiply(a_vec-a_hat_t, x_t) + w_t + u_vec
        x_sq = np.multiply(x_t,x_t)
        y_new = y_t + x_sq
        a_hat_new = np.divide(np.multiply(a_hat_t, y_t) + np.multiply(a_vec, x_sq) + np.multiply(w_t,x_t), y_new)
        
        # Add to trace
        x_trace.append(x_new)
        y_trace.append(y_new)
        a_hat_trace.append(a_hat_new)
        w_trace.append(w_t)
    return x_trace, a_hat_trace

def make_time_histogram(data, num_bins, center, reach):
    hists = []
    window = (center-reach, center+reach)
    for time_slice in data:
        bin_count, bin_labels = np.histogram(time_slice, num_bins, range=window)
        hists.append(bin_count)
    return hists, bin_labels

def plot_time_histogram(hist, bins, brightness):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.power(hist, brightness))
    num_bins = len(bins) - 1
    loc = range(num_bins)[::num_bins//5]
    labels = bins[::num_bins//5]
    plt.xticks(loc, labels)
    
def plot_hist_on_sub(sub, hist, bins, brightness, D):
    sub.imshow(np.power(hist, brightness),aspect=4)
    num_bins = len(bins) - 1
    loc = range(num_bins)[::num_bins//5]
    labels = bins[::num_bins//5]
    sub.set_xticks(loc)
    sub.set_xticklabels(labels)
    plt.setp(sub.get_xticklabels(), rotation=30, horizontalalignment='right')

W_P = (0, 1)
a_n, u_n, x0_n = 7, 7, 7
D = 1

f, ax = plt.subplots(a_n, 2*u_n, figsize=(30, 20))
for x0 in np.logspace(-1, 2, x0_n):
    X0_P = (x0, 0)
    for a, i in zip(np.logspace(-1, 2, a_n), range(a_n)):
        for u, j in zip(np.logspace(-1, 2, u_n), range(u_n)):
            x_tr, ah_tr = trials(a, u, W_P, X0_P, 50000, 30)
            #x_lim = np.max(np.abs(x_tr))
            #ah_lim = np.max(np.abs(ah_tr))
            x_lim = 3*np.var(x_tr[-1])
            ah_lim = 3*np.var(ah_tr[-1])
            xh, x_bins = make_time_histogram(x_tr, 200, u, x_lim)
            ah, a_bins = make_time_histogram(ah_tr, 200, a, ah_lim)
            ax[i, 2*j].set_title('x:u={0}\na={1}'.format(np.around(u, decimals=3), np.around(a,decimals=3)))
            ax[i, 2*j+1].set_title('a^:u={0}\na={1}'.format(np.around(u, decimals=3), np.around(a,decimals=3)))
            plot_hist_on_sub(ax[i,2*j], xh, np.around(x_bins, decimals=3), 0.5, D)
            plot_hist_on_sub(ax[i,2*j+1], ah, np.around(a_bins, decimals=3), 0.5, D)
    f.savefig('x0={0}.png'.format(x0))
