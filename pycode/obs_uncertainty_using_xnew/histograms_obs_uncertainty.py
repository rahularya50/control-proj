import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

np.random.seed()

def trials(a, u, w_param, x0_param, N, T):
    mu_x, sigma_x = x0_param;
    mu_w, sigma_w = w_param;
    np.random.seed();
    # All traces have elements that are arrays full of values per trials
    # Observation noise, w ~ Normal(0, sigma_w^2)
    w_trace = [np.random.normal(loc=mu_w, scale=sigma_w, size=N)];
    # State, with x0 ~ Normal(mu_x, sigma_x^2)
    x_trace = [np.random.normal(loc=mu_x, scale=sigma_x, size=N)];
    # Estimate of initial state
    x0_hat_trace = [np.zeros(N)];
    # Estimate of state trace
    x_hat_trace = [np.zeros(N)];
    # Observations y = x + w
    y_trace = [x_trace[-1] + w_trace[-1]];
    # Estimated a
    a_hat_trace = [np.zeros(N)];
    # Extra state defined as z_t = sum(y_t^2)
    z_trace = [np.zeros(N)];
    a_vec = np.repeat(a, N);
    u_vec = np.repeat(u, N);
    
    for i in range(T-1):
        # Get current state(s)
        x_t = x_trace[-1]
        x0_hat_t = x0_hat_trace[-1]
        x_hat_t = x_hat_trace[-1]
        z_t = z_trace[-1]
        a_hat_t = a_hat_trace[-1]
        y_t = y_trace[-1]
        w_t = np.random.normal(loc=mu_w, scale=sigma_w, size=N)
        
        u_t = -np.multiply(a_hat_t, x_hat_t) + u_vec
        # State update
        x_new = np.multiply(a_vec, x_t) + u_t
        # Observation update
        y_new = x_new + w_t 
        # Compute ahat, xhat0, xhat, then the control
        y_sq = np.multiply(y_t,y_t)
        z_new = z_t + y_sq
        
        a_hat_new = np.divide(np.multiply(a_hat_t, z_t) + np.multiply(y_new - u_t, y_t), z_new)
        #print(a_hat_new) 
        #x0_hat_new = TODO
        #x_hat_new = y_new
        x_hat_new = x_new

        # Add to trace
        x_trace.append(x_new)
        #x0_hat_trace.append(x0_hat_new)
        x_hat_trace.append(x_hat_new)
        y_trace.append(y_new)
        z_trace.append(z_new)
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

# Use the aspect argument to change aspect ratio
def plot_hist_on_sub(sub, hist, bins, brightness):
    sub.imshow(np.power(hist, brightness),aspect=1)
    num_bins = len(bins) - 1
    loc = range(num_bins)[::num_bins//5]
    labels = bins[::num_bins//5]
    sub.set_xticks(loc)
    sub.set_xticklabels(labels)
    plt.setp(sub.get_xticklabels(), rotation=30, horizontalalignment='right')

a_n, u_n, x_s_n = 7, 7, 7

for a in np.logspace(-1, 2, a_n):
    f, ax = plt.subplots(u_n, 2*x_s_n, figsize=(40,16))
    for u, i in zip(np.logspace(-1, 2, u_n), range(u_n)):
        for x_s, j in zip(np.logspace(-1, 2, x_s_n), range(x_s_n)):
            x_tr, ah_tr = trials(a, u, (0, 1),(1, 0), 20000, 50)
            x_std = np.sqrt(np.var(x_tr[-1]))
            ah_std = np.sqrt(np.var(ah_tr[-1]))
            xh, x_bins = make_time_histogram(x_tr, 100, u, 3*x_std)
            ah, a_bins = make_time_histogram(ah_tr, 100, a, 3*ah_std)
            ax[i, 2*j].set_title('x:u={0}'.format(np.around(u, decimals=3)))
            ax[i, 2*j+1].set_title('a^:u={0}'.format(np.around(u, decimals=3)))
            plot_hist_on_sub(ax[i,2*j], xh, np.around(x_bins, decimals=3), 0.5)
            plot_hist_on_sub(ax[i,2*j+1], ah, np.around(a_bins, decimals=3), 0.5)
    f.savefig('a={0}.png'.format(a))
