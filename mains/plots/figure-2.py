import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/N-100_N0-500-1000/eval_data/'
set_metrics_100 = np.load(path + 'N100-set.npz')['metrics']
path = '/N/slate/cwseitz/cvdm/Sim/4x/N-200_N0-500-1000/eval_data/'
set_metrics_200 = np.load(path + 'N200-set.npz')['metrics']
path = '/N/slate/cwseitz/cvdm/Sim/4x/N-500_N0-500-1000/eval_data/'
set_metrics_500 = np.load(path + 'N500-set.npz')['metrics']

print(set_metrics_100.shape)

metrics_100 = set_metrics_100.reshape(-1, 4)
metrics_200 = set_metrics_200.reshape(-1, 4)
metrics_500 = set_metrics_500.reshape(-1, 4)

def compute_precision_recall(metrics):
    intersection = metrics[:, 0]
    union = metrics[:, 1]
    false_positive = metrics[:, 2]
    false_negative = metrics[:, 3]
    
    precision = intersection / (intersection + false_positive)
    recall = intersection / (intersection + false_negative)
    
    return precision, recall

precision_100, recall_100 = compute_precision_recall(metrics_100)
precision_200, recall_200 = compute_precision_recall(metrics_200)
precision_500, recall_500 = compute_precision_recall(metrics_500)

def compute_mean_std(data):
    return np.mean(data), np.std(data)

metrics_dict = {
    '100': (precision_100, recall_100),
    '200': (precision_200, recall_200),
    '500': (precision_500, recall_500)
}

densities = ['100', '200', '500']
density_values = [100, 200, 500]
precision_means = []
precision_stds = []
recall_means = []
recall_stds = []

for density in densities:
    precision, recall = metrics_dict[density]
    p_mean, p_std = compute_mean_std(precision)
    r_mean, r_std = compute_mean_std(recall)
    
    precision_means.append(p_mean)
    precision_stds.append(p_std)
    recall_means.append(r_mean)
    recall_stds.append(r_std)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4),sharey=True)

ax1.errorbar(density_values, precision_means, yerr=precision_stds, fmt='x', 
             capsize=5, capthick=1, label='Precision', color='black')
ax1.set_xlabel(r'$\rho$',fontsize=16)
ax1.set_ylabel('Precision',fontsize=16)
ax1.grid()
ax1.set_xticks(density_values)

ax2.errorbar(density_values, recall_means, yerr=recall_stds, fmt='x', 
             capsize=5, capthick=1, label='Recall', color='black')
ax2.set_xlabel(r'$\rho$',fontsize=16)
ax2.set_ylabel('Recall',fontsize=16)
ax2.grid()
ax2.set_xticks(density_values)

plt.tight_layout()
plt.savefig('/N/slate/cwseitz/cvdm/Sim/4x/Figure-2.png',dpi=300)
plt.show()

